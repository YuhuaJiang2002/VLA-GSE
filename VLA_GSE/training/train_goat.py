# Copyright VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 


"""
VLA-GSE GOAT trainer for Multi-GPU (No DeepSpeed).
Based on train_lora_gpu1.py, but uses GOAT (Gated MoE LoRA) instead of standard LoRA.

Key Features:
  - Load pretrained base_vlm
  - Apply GOAT (Mixture of Experts LoRA) to the VLM backbone
  - Keep action_model fully trainable
  - Finetune with GOAT + action head for efficient adaptation
  - Multi-GPU training without DeepSpeed
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
from accelerate import Accelerator
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# GOAT-PEFT
from VLA_GSE.goat_peft import GOATConfig, GOATModel, get_goat_model
from VLA_GSE.goat_peft.utils.peft_types import TaskType

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
accelerator = Accelerator()


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    accelerator.wait_for_everyone()

    return output_dir


def apply_goat_to_vlm(model, cfg):
    """
    Apply GOAT (Gated MoE LoRA) to the VLM backbone while keeping action_model fully trainable.
    
    Args:
        model: The full VLA model (e.g., Qwenvl_OFT)
        cfg: Configuration object containing GOAT settings
    
    Returns:
        model: Model with GOAT applied to VLM backbone
    """
    # Get GOAT config from yaml or use defaults
    goat_cfg = cfg.trainer.get("goat", {})
    if not goat_cfg:
        goat_cfg = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05,
            "bias": "none",
            "num_experts": 8,
            "top_k": 2,
            "init_type": "goat",
            "init_cof": 1.0,
        }
        logger.info("Using default GOAT configuration")
    
    logger.info(f"Applying GOAT with config: r={goat_cfg.get('r', 16)}, "
                f"num_experts={goat_cfg.get('num_experts', 8)}, top_k={goat_cfg.get('top_k', 2)}")
    
    # Find the VLM model within the framework
    vlm_model = None
    vlm_attr_path = None
    
    # Try different common paths to find the VLM model
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
        logger.warning("Could not find VLM model to apply GOAT. Training all VLM parameters instead.")
        return model
    
    logger.info(f"Found VLM model at path: {vlm_attr_path}")
    
    # Create GOAT configuration
    goat_config = GOATConfig(
        r=goat_cfg.get("r", 16),
        lora_alpha=goat_cfg.get("lora_alpha", 32),
        target_modules=goat_cfg.get("target_modules", "all-linear"),
        lora_dropout=goat_cfg.get("lora_dropout", 0.05),
        bias=goat_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        num_experts=goat_cfg.get("num_experts", 8),
        top_k=goat_cfg.get("top_k", 2),
        init_type=goat_cfg.get("init_type", "goat"),
        init_cof=goat_cfg.get("init_cof", 1.0),
    )
    
    # Apply GOAT to the VLM model
    goat_vlm_model = get_goat_model(vlm_model, goat_config)
    
    # Replace the original VLM model with the GOAT model
    if vlm_attr_path == "qwen_vl_interface.model":
        model.qwen_vl_interface.model = goat_vlm_model
    elif vlm_attr_path == "vlm":
        model.vlm = goat_vlm_model
    elif vlm_attr_path == "qwen_vl.model":
        model.qwen_vl.model = goat_vlm_model
    
    # Print trainable parameters from GOAT model
    goat_vlm_model.print_trainable_parameters()
    
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
    """build model framework with GOAT applied to VLM"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    
    # First build the full framework
    model = build_framework(cfg)
    
    # Apply GOAT to VLM backbone
    model = apply_goat_to_vlm(model, cfg)
    
    # Ensure action_model is trainable
    model = ensure_action_model_trainable(model)
    
    return model


def prepare_data(cfg) -> DataLoader:
    """prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler for GOAT training"""
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


def freeze_vlm_non_goat_params(model):
    """
    Freeze VLM parameters that don't contain GOAT/LoRA.
    Only GOAT parameters (containing 'lora' in name) and action_model parameters remain trainable.
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
        
        logger.info(f"🔒 VLM ({vlm_path}): Frozen {frozen_count / 1e6:.3f}M non-GOAT params, "
                    f"Trainable {trainable_count / 1e6:.3f}M GOAT params")
    else:
        logger.warning("⚠️ Could not find VLM model to freeze non-GOAT parameters")
    
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


class AccelerateGOATTrainer:
    """Trainer for GOAT-based VLA finetuning on Multi-GPU"""
    
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, device, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = cfg.datasets.vla_data.per_device_batch_size * self.accelerator.num_processes
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(cfg.trainer, "gradient_accumulation_steps", 1)
        
        # GOAT aux loss weight
        self.aux_loss_weight = getattr(cfg.trainer.goat, "aux_loss_weight", 0.01) if hasattr(cfg.trainer, "goat") else 0.01
    
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

        # Freeze VLM non-GOAT parameters
        self.model = freeze_vlm_non_goat_params(self.model)
        
        # Apply any additional freezing from config
        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        if freeze_modules:
            self.model = freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Print trainable parameters
        print_trainable_parameters(self.model)

        if self.accelerator.is_main_process:
            logger.info(f"✅ Model will be placed on {self.device} by Accelerate")

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
            logger.info("No pretrained checkpoint provided. Starting GOAT training from scratch.")
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
        if self.accelerator.is_main_process:
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)

            logger.info(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics"""
        if self.accelerator.is_main_process and self.completed_steps % self.config.trainer.logging_frequency == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics.get('action_loss', 'N/A'):.4f}, "
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
        
        progress_bar = tqdm(range(self.config.trainer.max_train_steps), desc="Training", disable=not self.accelerator.is_local_main_process)
        
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
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                avg_loss = accumulation_loss / self.gradient_accumulation_steps
                avg_aux_loss = accumulation_aux_loss / self.gradient_accumulation_steps
                accumulation_loss = 0.0
                accumulation_aux_loss = 0.0
                
                self.completed_steps += 1
                if self.accelerator.is_local_main_process:
                    progress_bar.update(1)

                    progress_bar.set_postfix({
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
            output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)

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
        if not self.accelerator.is_main_process:
            return
        logger.info("***** GOAT Multi-GPU Training Configuration *****")
        logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
        logger.info(f"  Batch size = {self.config.datasets.vla_data.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.total_batch_size}")
        logger.info(f"  Device = {self.device}")
        
        # Log GOAT config if available
        goat_cfg = self.config.trainer.get("goat", {})
        if goat_cfg:
            logger.info(f"  GOAT rank = {goat_cfg.get('r', 16)}")
            logger.info(f"  GOAT num_experts = {goat_cfg.get('num_experts', 8)}")
            logger.info(f"  GOAT top_k = {goat_cfg.get('top_k', 2)}")
            logger.info(f"  GOAT init_type = {goat_cfg.get('init_type', 'goat')}")

    def _train_step(self, batch_vla):
        """Execute single training step"""
        # VLA task forward propagation with mixed precision
        with autocast(dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            
            # Get GOAT auxiliary loss (load balancing)
            aux_loss = torch.tensor(0.0, device=self.device)
            try:
                # Try to get aux loss from GOAT model
                vlm_model = None
                base_model = self.accelerator.unwrap_model(self.model)
                if hasattr(base_model, "qwen_vl_interface") and hasattr(base_model.qwen_vl_interface, "model"):
                    vlm_model = base_model.qwen_vl_interface.model
                elif hasattr(base_model, "vlm"):
                    vlm_model = base_model.vlm
                
                if vlm_model is not None and hasattr(vlm_model, "get_aux_loss"):
                    aux_loss = vlm_model.get_aux_loss()
            except Exception:
                pass
            
            # Total loss
            total_loss = action_loss + self.aux_loss_weight * aux_loss
            
            # Scale loss for gradient accumulation
            loss = total_loss / self.gradient_accumulation_steps

        # Backward propagation with scaler
        self.accelerator.backward(loss)

        return action_loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

    def _finalize_training(self):
        """Training end processing"""
        final_checkpoint = os.path.join(self.config.output_dir, "final_goat_model")
        if self.accelerator.is_main_process:
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final GOAT model saved at {final_checkpoint}")
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA GOAT Multi-GPU Training :: Warming Up")
    
    # Set device
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    
    # Create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    
    # Build model with GOAT applied
    vla = build_model(cfg)
    
    # Prepare data
    vla_train_dataloader = prepare_data(cfg=cfg)

    # Set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # Create trainer
    trainer = AccelerateGOATTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        accelerator=accelerator,
    )

    # Execute training preparation before distributed wrapping.
    trainer.prepare_training()
    trainer.model, trainer.optimizer, trainer.vla_train_dataloader, trainer.lr_scheduler = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.vla_train_dataloader, trainer.lr_scheduler
    )

    # Execute training
    trainer.train()

    logger.info("GOAT training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="VLA_GSE/config/training/vla_gse_cotrain_oxe.yaml", help="Path to YAML config")
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
