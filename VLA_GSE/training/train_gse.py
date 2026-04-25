# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 


"""
StarVLA's GSE (Generalized and Specialized Expert) trainer for Single GPU (No DeepSpeed).
Based on train_goat.py, but uses GSE instead of standard GOAT.

Key Features:
  - Load pretrained base_vlm
  - Apply GSE (Generalized and Specialized Expert MoE LoRA) to the VLM backbone
  - Generalized experts are always activated (not routed)
  - Specialized experts are selected by top-k router
  - Keep action_model fully trainable
  - Single GPU training without DeepSpeed
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re
import logging

# Third-Party Libraries
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# GSE-PEFT
from VLA_GSE.gse_peft import GSEConfig, GSEModel, get_gse_model
from VLA_GSE.gse_peft.utils.peft_types import TaskType

# Local Modules
from VLA_GSE.training.trainer_utils.trainer_tools import normalize_dotlist_args
from VLA_GSE.model.framework import build_framework
from VLA_GSE.training.trainer_utils.trainer_tools import build_param_lr_groups
from VLA_GSE.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from VLA_GSE.dataloader import build_dataloader

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def apply_gse_to_vlm(model, cfg, skip_svd=False, gse_cache_path=None):
    """
    Apply GSE (Generalized and Specialized Expert MoE LoRA) to the VLM backbone
    while keeping action_model fully trainable.

    Args:
        model: The full VLA model (e.g., Qwenvl_OFT)
        cfg: Configuration object containing GSE settings
        skip_svd: If True, skip SVD computation during init.
        gse_cache_path: If provided, cache the post-SVD state_dict for reuse.

    Returns:
        model: Model with GSE applied to VLM backbone
    """
    gse_cfg = cfg.trainer.get("gse", {})
    if not gse_cfg:
        gse_cfg = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05,
            "bias": "none",
            "num_experts": 8,
            "num_generalized_experts": 2,
            "top_k": 2,
            "init_type": "gse",
            "init_cof": 1.0,
        }
        logger.info("Using default GSE configuration")

    init_type = gse_cfg.get("init_type", "gse")

    logger.info(f"Applying GSE with config: r={gse_cfg.get('r', 16)}, "
                f"num_experts={gse_cfg.get('num_experts', 8)}, "
                f"num_generalized_experts={gse_cfg.get('num_generalized_experts', 2)}, "
                f"top_k={gse_cfg.get('top_k', 2)}, "
                f"init_type={init_type}, skip_svd={skip_svd}")

    vlm_model = None
    vlm_attr_path = None

    if hasattr(model, "qwen_vl_interface") and hasattr(model.qwen_vl_interface, "model"):
        vlm_model = model.qwen_vl_interface.model
        vlm_attr_path = "qwen_vl_interface.model"
    elif hasattr(model, "vlm"):
        vlm_model = model.vlm
        vlm_attr_path = "vlm"
    elif hasattr(model, "qwen_vl") and hasattr(model.qwen_vl, "model"):
        vlm_model = model.qwen_vl.model
        vlm_attr_path = "qwen_vl.model"

    if vlm_model is None:
        logger.warning("Could not find VLM model to apply GSE. Training all VLM parameters instead.")
        return model

    logger.info(f"Found VLM model at path: {vlm_attr_path}")

    gse_config = GSEConfig(
        r=gse_cfg.get("r", 16),
        lora_alpha=gse_cfg.get("lora_alpha", 32),
        target_modules=gse_cfg.get("target_modules", "all-linear"),
        lora_dropout=gse_cfg.get("lora_dropout", 0.05),
        bias=gse_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        num_experts=gse_cfg.get("num_experts", 8),
        num_generalized_experts=gse_cfg.get("num_generalized_experts", 2),
        top_k=gse_cfg.get("top_k", 2),
        init_type=init_type,
        init_cof=gse_cfg.get("init_cof", 1.0),
        skip_svd_init=skip_svd,
        aux_loss_weight=float(gse_cfg.get("aux_loss_weight", 0.01)),
    )

    gse_vlm_model = get_gse_model(
        vlm_model, gse_config, gse_cache_path=gse_cache_path)

    if vlm_attr_path == "qwen_vl_interface.model":
        model.qwen_vl_interface.model = gse_vlm_model
    elif vlm_attr_path == "vlm":
        model.vlm = gse_vlm_model
    elif vlm_attr_path == "qwen_vl.model":
        model.qwen_vl.model = gse_vlm_model

    gse_vlm_model.print_trainable_parameters()

    if hasattr(gse_vlm_model, "get_expert_info"):
        expert_info = gse_vlm_model.get_expert_info()
        logger.info(f"GSE Expert Info: {expert_info['total_generalized_experts']} generalized, "
                   f"{expert_info['total_specialized_experts']} specialized experts total")

    return model

def ensure_action_model_trainable(model):
    """
    Ensure the action_model parameters are trainable.
    
    Args:
        model: The full VLA model
    
    Returns:
        model: Model with action_model parameters set to trainable
    """
    action_model_paths = ["action_model", "action_head", "action_decoder"]
    
    for path in action_model_paths:
        if hasattr(model, path):
            action_module = getattr(model, path)
            trainable_count = 0
            for param in action_module.parameters():
                param.requires_grad = True
                trainable_count += param.numel()
            logger.info(f"Set {path} to trainable: {trainable_count:,} parameters")
            break
    
    # Also make action_query trainable if exists (for QwenOFT)
    if hasattr(model, "action_query"):
        model.action_query.requires_grad = True
        logger.info(f"Set action_query to trainable: {model.action_query.numel():,} parameters")
    
    return model


def build_model(cfg) -> torch.nn.Module:
    """build model framework with GSE applied to VLM"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    
    # First build the full framework
    model = build_framework(cfg)
    
    # Apply GSE to VLM backbone
    model = apply_gse_to_vlm(model, cfg)
    
    # Ensure action_model is trainable
    model = ensure_action_model_trainable(model)
    
    return model


def prepare_data(cfg) -> DataLoader:
    """prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler for GSE training"""
    # Build parameter groups - only trainable parameters will be included
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    for i, group in enumerate(optimizer.param_groups):
        logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


def print_trainable_parameters(model):
    """Print the total number of parameters and trainable parameters"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model parameters: {num_params / 1e6:.3f}M Total, {num_trainable_params / 1e6:.3f}M Trainable")
    return num_params, num_trainable_params


def freeze_vlm_non_gse_params(model):
    """
    Freeze VLM parameters that don't contain GSE/LoRA.
    Only GSE parameters (containing 'lora' in name) and action_model parameters remain trainable.
    """
    frozen_count = 0
    trainable_count = 0
    
    # Find the VLM model
    vlm_model = None
    vlm_path = None
    
    if hasattr(model, "qwen_vl_interface") and hasattr(model.qwen_vl_interface, "model"):
        vlm_model = model.qwen_vl_interface.model
        vlm_path = "qwen_vl_interface.model"
    elif hasattr(model, "vlm"):
        vlm_model = model.vlm
        vlm_path = "vlm"
    elif hasattr(model, "qwen_vl") and hasattr(model.qwen_vl, "model"):
        vlm_model = model.qwen_vl.model
        vlm_path = "qwen_vl.model"
    
    if vlm_model is not None:
        # Freeze all VLM parameters that don't contain 'lora' in their name
        for name, param in vlm_model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        
        logger.info(f"🔒 VLM ({vlm_path}): Frozen {frozen_count / 1e6:.3f}M non-GSE params, "
                    f"Trainable {trainable_count / 1e6:.3f}M GSE params")
    else:
        logger.warning("⚠️ Could not find VLM model to freeze non-GSE parameters")
    
    return model


def freeze_backbones(model, freeze_modules=""):
    """Freeze specified modules (legacy function, kept for compatibility)"""
    frozen = []
    if freeze_modules and isinstance(freeze_modules, str):
        patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
        for path in patterns:
            attrs = path.split(".")
            module = model
            try:
                for attr in attrs:
                    module = getattr(module, attr)
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(path)
            except AttributeError:
                logger.warning(f"⚠️ Module path does not exist, cannot freeze: {path}")
                continue
    if frozen:
        logger.info(f"🔒 Frozen modules: {frozen}")
    return model


class SingleGPUGSETrainer:
    """Trainer for GSE-based VLA finetuning on Single GPU"""
    
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, device):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.completed_steps = 0
        self.total_batch_size = cfg.datasets.vla_data.per_device_batch_size
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(cfg.trainer, "gradient_accumulation_steps", 1)
        
        # GSE aux loss weight (only for specialized experts)
        self.aux_loss_weight = getattr(cfg.trainer.gse, "aux_loss_weight", 0.01) if hasattr(cfg.trainer, "gse") else 0.01
    
    def prepare_training(self):
        """Prepare training"""
        # Set seed
        seed = getattr(self.config, "seed", 3047)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize checkpointing
        self._init_checkpointing()
        
        # Adjust LR scheduler for resume
        self._adjust_lr_scheduler_for_resume()

        # Freeze VLM non-GSE parameters
        self.model = freeze_vlm_non_gse_params(self.model)
        
        # Apply any additional freezing from config
        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        if freeze_modules:
            self.model = freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Print trainable parameters
        print_trainable_parameters(self.model)

        # Move model to GPU
        self.model = self.model.to(self.device)
        logger.info(f"✅ Model moved to {self.device}")

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler for resume from checkpoint"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        
        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self._load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}, steps: {self.completed_steps}")
                return
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0
        
        if pretrained_checkpoint:
            self._load_checkpoint(pretrained_checkpoint)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except AttributeError:
                logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting GSE training from scratch.")
            self.completed_steps = 0

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Get the latest checkpoint from directory"""
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith("_pytorch_model.pt")]
        if not checkpoints:
            return None, 0
        
        # Extract step numbers and find the latest
        steps = []
        for ckpt in checkpoints:
            match = re.search(r"steps_(\d+)_pytorch_model\.pt", ckpt)
            if match:
                steps.append((int(match.group(1)), ckpt))
        
        if not steps:
            return None, 0
        
        latest_step, latest_ckpt = max(steps, key=lambda x: x[0])
        return os.path.join(checkpoint_dir, latest_ckpt), latest_step

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=False)
        logger.info("✅ Checkpoint loaded")

    def _save_checkpoint(self):
        """Save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        
        # Save model state
        state_dict = self.model.state_dict()
        torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

        # Save training metadata
        summary_data = {"steps": self.completed_steps}
        with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary_data) + "\n")
        
        # Save accessed configuration
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        
        logger.info(f"✅ Checkpoint saved at {checkpoint_path}")

    def _log_metrics(self, metrics):
        """Record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = self.vla_epoch_count
            logger.info(f"Step {self.completed_steps}, Epoch {self.vla_epoch_count}, "
                       f"Loss: {metrics.get('action_loss', 'N/A'):.4f}, "
                       f"Aux Loss: {metrics.get('aux_loss', 0):.4f}, LR: {metrics['learning_rate']:.2e}")

    def _create_data_iterators(self):
        """Create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_epoch_count = 0

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            logger.info(f"Epoch {self.vla_epoch_count} completed, starting new epoch...")
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)
        return batch_vla

    def train(self):
        """Execute training loop"""
        self._log_training_config()
        self._create_data_iterators()
        
        self.model.train()
        
        progress_bar = tqdm(range(self.config.trainer.max_train_steps), desc="Training")
        
        accumulation_loss = 0.0
        accumulation_aux_loss = 0.0

        while self.completed_steps < self.config.trainer.max_train_steps:
            # Get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # Execute training step
            t_start_model = time.perf_counter()
            step_loss, aux_loss = self._train_step(batch_vla)
            accumulation_loss += step_loss
            accumulation_aux_loss += aux_loss
            t_end_model = time.perf_counter()

            # Update progress (only after gradient accumulation)
            if (self.completed_steps + 1) % self.gradient_accumulation_steps == 0 or self.gradient_accumulation_steps == 1:
                # Gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                avg_loss = accumulation_loss / self.gradient_accumulation_steps
                avg_aux_loss = accumulation_aux_loss / self.gradient_accumulation_steps
                accumulation_loss = 0.0
                accumulation_aux_loss = 0.0
                
                self.completed_steps += 1
                progress_bar.update(1)
                
                progress_bar.set_postfix({
                    "epoch": self.vla_epoch_count,
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "model": f"{t_end_model - t_start_model:.3f}s",
                    "loss": f"{avg_loss:.4f}",
                    "aux": f"{avg_aux_loss:.4f}"
                })

                # Record metrics
                step_metrics = {
                    "action_loss": avg_loss,
                    "aux_loss": avg_aux_loss,
                    "data_time": t_end_data - t_start_data,
                    "model_time": t_end_model - t_start_model
                }

                # Evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0 and self.completed_steps > 0:
                    step_metrics = self.eval_action_model(step_metrics)

                self._log_metrics(step_metrics)

                # Save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            # Check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Evaluate the model on a batch"""
        self.model.eval()
        with torch.no_grad():
            examples = self._get_next_batch()
            actions = [example["action"] for example in examples]
            
            # Predict actions using the model
            output_dict = self.model.predict_action(examples=examples)

            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            
            # Match action chunk length
            chunk_len = normalized_actions.shape[1]
            actions_target = actions[:, -chunk_len:, :]
            num_pots = np.prod(actions_target.shape)
            
            score = np.linalg.norm(normalized_actions - actions_target)
            step_metrics["mse_score"] = score / num_pots

            del examples
        
        self.model.train()
        return step_metrics

    def _log_training_config(self):
        """Record training config"""
        logger.info("***** GSE (Generalized and Specialized Expert) Single GPU Training Configuration *****")
        logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
        logger.info(f"  Batch size = {self.config.datasets.vla_data.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.config.datasets.vla_data.per_device_batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  Device = {self.device}")
        
        # Log GSE config if available
        gse_cfg = self.config.trainer.get("gse", {})
        if gse_cfg:
            logger.info(f"  GSE rank = {gse_cfg.get('r', 16)}")
            logger.info(f"  GSE num_experts = {gse_cfg.get('num_experts', 8)}")
            logger.info(f"  GSE num_generalized_experts = {gse_cfg.get('num_generalized_experts', 2)}")
            logger.info(f"  GSE top_k = {gse_cfg.get('top_k', 2)}")
            logger.info(f"  GSE init_type = {gse_cfg.get('init_type', 'gse')}")

    def _train_step(self, batch_vla):
        """Execute single training step"""
        # VLA task forward propagation with mixed precision (bf16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            
            # Get GSE auxiliary loss (load balancing for specialized experts only)
            aux_loss = torch.tensor(0.0, device=self.device)
            try:
                # Try to get aux loss from GSE model
                vlm_model = None
                if hasattr(self.model, "qwen_vl_interface") and hasattr(self.model.qwen_vl_interface, "model"):
                    vlm_model = self.model.qwen_vl_interface.model
                elif hasattr(self.model, "vlm"):
                    vlm_model = self.model.vlm
                
                if vlm_model is not None and hasattr(vlm_model, "get_aux_loss"):
                    aux_loss = vlm_model.get_aux_loss()
            except Exception:
                pass
            
            # Total loss
            total_loss = action_loss + self.aux_loss_weight * aux_loss
            
            # Scale loss for gradient accumulation
            loss = total_loss / self.gradient_accumulation_steps

        # Backward propagation (no scaler needed for bf16)
        loss.backward()

        return action_loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

    def _finalize_training(self):
        """Training end processing"""
        final_checkpoint = os.path.join(self.config.output_dir, "final_gse_model")
        os.makedirs(final_checkpoint, exist_ok=True)
        
        # Save full model state
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
        
        logger.info(f"Training complete. Final GSE model saved at {final_checkpoint}")


def main(cfg) -> None:
    logger.info("VLA GSE (Generalized and Specialized Expert) Single GPU Training :: Warming Up")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    
    # Create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    
    # Build model with GSE applied
    vla = build_model(cfg)
    
    # Prepare data
    vla_train_dataloader = prepare_data(cfg=cfg)

    # Set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # Create trainer
    trainer = SingleGPUGSETrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    # Execute training preparation
    trainer.prepare_training()
    
    # Execute training
    trainer.train()

    logger.info("GSE training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="VLA_GSE/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Debug mode support
    if getattr(cfg, "is_debug", False):
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
