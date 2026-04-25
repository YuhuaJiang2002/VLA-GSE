# Copyright VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 


"""
Configurable PEFT trainer.
Based on the LoRA trainer, but exposes a single method selector for LoRA-family
baselines while keeping the action head fully trainable.

Key Features:
  - Load pretrained base_vlm
  - Apply a selected PEFT method to the VLM backbone
  - Keep action_model fully trainable (not using LoRA)
  - Finetune with PEFT + action head for efficient adaptation
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

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

# Local Modules
from VLA_GSE.training.trainer_utils.trainer_tools import normalize_dotlist_args
from VLA_GSE.model.framework import build_framework
from VLA_GSE.training.trainer_utils.trainer_tools import TrainerUtils
from VLA_GSE.training.trainer_utils.trainer_tools import build_param_lr_groups
from VLA_GSE.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from VLA_GSE.dataloader import build_dataloader

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Logger
logger = get_logger(__name__)


# Helper functions for single/multi-GPU compatibility
def is_distributed():
    """Check if distributed training is initialized"""
    return dist.is_initialized()


def get_rank():
    """Get current process rank, returns 0 when distributed is not initialized"""
    return dist.get_rank() if is_distributed() else 0


def is_main_process():
    """Check if current process is the main process"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes, no-op when distributed is not initialized"""
    if is_distributed():
        dist.barrier()


def destroy_process_group():
    """Destroy process group if distributed training is initialized"""
    if is_distributed():
        dist.destroy_process_group()


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


SUPPORTED_OTHER_PEFT_METHODS = (
    "lora",
    "rslora",
    "dora",
    "pissa",
    "molora",
    "adamole",
    "hydralora",
    "milora",
)


def _get_other_peft_cfg(cfg):
    """Read configurable PEFT settings while preserving LoRA defaults."""
    peft_cfg = cfg.trainer.get("other_peft", {})
    if not peft_cfg:
        peft_cfg = cfg.trainer.get("lora", {})

    if not peft_cfg:
        peft_cfg = {
            "method": "lora",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05,
            "bias": "none",
        }
        logger.info("Using default PEFT configuration")

    method = str(peft_cfg.get("method", "lora")).lower()
    if method not in SUPPORTED_OTHER_PEFT_METHODS:
        raise ValueError(
            f"Unsupported PEFT method '{method}'. "
            f"Choose one of: {', '.join(SUPPORTED_OTHER_PEFT_METHODS)}"
        )
    return peft_cfg, method


def _build_lora_family_config(peft_cfg, method):
    """Build a PEFT LoraConfig for LoRA-family baselines."""
    config_kwargs = {
        "r": peft_cfg.get("r", 16),
        "lora_alpha": peft_cfg.get("lora_alpha", 32),
        "target_modules": peft_cfg.get("target_modules", "all-linear"),
        "lora_dropout": peft_cfg.get("lora_dropout", 0.05),
        "bias": peft_cfg.get("bias", "none"),
        "task_type": TaskType.CAUSAL_LM,
    }

    if method == "rslora":
        config_kwargs["use_rslora"] = True
    elif method == "dora":
        config_kwargs["use_dora"] = True
    elif method == "pissa":
        config_kwargs["init_lora_weights"] = peft_cfg.get("init_lora_weights", "pissa")
    elif method == "milora":
        config_kwargs["init_lora_weights"] = peft_cfg.get("init_lora_weights", "pissa")
        logger.warning(
            "MiLoRA is selected through the shared PEFT baseline entry point. "
            "PEFT does not expose a native MiLoRA initializer in this environment, "
            "so the script uses a PiSSA-style spectral initializer by default."
        )
    elif method in {"molora", "adamole", "hydralora"}:
        logger.warning(
            "%s is selected through the shared PEFT baseline entry point. "
            "This script uses the LoRA-compatible PEFT adapter unless a custom "
            "implementation is wired in through the config.",
            method,
        )

    return LoraConfig(**config_kwargs)


def apply_other_peft_to_vlm(model, cfg):
    """
    Apply the selected PEFT method to the VLM backbone while keeping action_model
    fully trainable.
    
    Args:
        model: The full VLA model (e.g., Qwenvl_OFT)
        cfg: Configuration object containing PEFT settings
    
    Returns:
        model: Model with PEFT applied to VLM backbone
    """
    peft_cfg, method = _get_other_peft_cfg(cfg)
    logger.info(
        "Applying %s with config: r=%s, alpha=%s",
        method,
        peft_cfg.get("r", 16),
        peft_cfg.get("lora_alpha", 32),
    )
    
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
        logger.warning("Could not find VLM model to apply PEFT. Training all VLM parameters instead.")
        return model
    
    logger.info(f"Found VLM model at path: {vlm_attr_path}")
    
    peft_config = _build_lora_family_config(peft_cfg, method)
    
    # Apply PEFT to the VLM model
    peft_vlm_model = get_peft_model(vlm_model, peft_config)
    
    # Replace the original VLM model with the PEFT model
    if vlm_attr_path == "qwen_vl_interface.model":
        model.qwen_vl_interface.model = peft_vlm_model
    elif vlm_attr_path == "vlm":
        model.vlm = peft_vlm_model
    elif vlm_attr_path == "qwen_vl.model":
        model.qwen_vl.model = peft_vlm_model
    
    # Print trainable parameters from PEFT model
    if is_main_process():
        peft_vlm_model.print_trainable_parameters()
    
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
    """build model framework with PEFT applied to VLM"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    
    # First build the full framework
    model = build_framework(cfg)
    
    # Apply selected PEFT method to VLM backbone
    model = apply_other_peft_to_vlm(model, cfg)
    
    # Ensure action_model is trainable
    model = ensure_action_model_trainable(model)
    
    return model


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    # dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler for PEFT training"""
    # Build parameter groups - only trainable parameters will be included
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if is_main_process():
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


class OtherPEFTVLATrainer(TrainerUtils):
    """Trainer for configurable PEFT-based VLA finetuning"""
    
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
    
    def prepare_training(self):
        rank = get_rank()
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Initialize checkpointing
        self._init_checkpointing()
        
        # Adjust LR scheduler for resume
        self._adjust_lr_scheduler_for_resume()

        # Apply any additional freezing from config (should not freeze PEFT or action_model)
        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        if freeze_modules:
            self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Print trainable parameters
        self.print_trainable_parameters(self.model)

        # Let Accelerate place and wrap the model, optimizer, and dataloader.
        self.model, self.optimizer, self.vla_train_dataloader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        logger.info(f"Accelerate training mode: {self.accelerator.num_processes} process(es) on {self.accelerator.device}")

        # self._init_wandb()

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler for resume from checkpoint"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """Calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-other-peft-train",
            )

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
                self.model = self.load_pretrained_backbones(self.model, resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}, steps: {self.completed_steps}")
                return
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0
        
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except AttributeError:
                logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting PEFT training from scratch.")
            self.completed_steps = 0

    def _save_checkpoint(self):
        """Save current training state"""
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            
            # Save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # Save training metadata
            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            
            # Save accessed configuration
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
            
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if is_main_process():
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Loss: {metrics.get('action_dit_loss', 'N/A')}")

    def _create_data_iterators(self):
        """Create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        return batch_vla

    def train(self):
        """Execute training loop"""
        self._log_training_config()
        self._create_data_iterators()
        
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), 
            disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            # Get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # Execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # Update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "model": f"{t_end_model - t_start_model:.3f}s",
                    "loss": f"{step_metrics.get('action_dit_loss', 0):.4f}"
                })

            # Evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0 and self.completed_steps > 0:
                step_metrics = self.eval_action_model(step_metrics)

            # Record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
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
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        
        # Predict actions using the model
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            
            # Match action chunk length
            chunk_len = normalized_actions.shape[1]
            actions_target = actions[:, -chunk_len:, :]
            num_pots = np.prod(actions_target.shape)
            
            score = TrainerUtils.euclidean_distance(normalized_actions, actions_target)
            step_metrics["mse_score"] = score / num_pots

        del examples
        barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Other-PEFT Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            
            peft_cfg, method = _get_other_peft_cfg(self.config)
            logger.info(f"  PEFT method = {method}")
            logger.info(f"  PEFT rank = {peft_cfg.get('r', 16)}")
            logger.info(f"  PEFT alpha = {peft_cfg.get('lora_alpha', 32)}")

    def _train_step(self, batch_vla):
        """Execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            # Backward propagation
            self.accelerator.backward(total_loss)

            # Gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

        return {"action_dit_loss": action_loss.item()}

    def _finalize_training(self):
        """Training end processing"""
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_other_peft_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            
            # Save full model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            
            logger.info(f"Training complete. Final PEFT model saved at {final_checkpoint}")
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Other-PEFT Multi-GPU Training :: Warming Up")
    
    # Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    
    # Create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    
    # Build model with selected PEFT method applied
    vla = build_model(cfg)
    
    # Prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # Set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # Create trainer
    trainer = OtherPEFTVLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # Execute training preparation
    trainer.prepare_training()
    
    # Execute training
    trainer.train()

    logger.info("Other-PEFT training finished.")
    barrier()
    destroy_process_group()


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
    if getattr(cfg, "is_debug", False) and is_main_process():
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
