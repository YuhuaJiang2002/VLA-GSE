# Copyright 2025 VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 


"""
VLA-GSE Full Fine-Tuning (FFT) trainer for Multi-GPU.
Based on train_lora_gpu1.py, but trains all model parameters without LoRA.

Key Features:
  - Load pretrained base_vlm
  - Train all model parameters (VLM backbone + action head)
  - Multi-GPU training without DeepSpeed
  - Mixed precision (bf16) training (no GradScaler needed for bf16)
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
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

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
    """Create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    accelerator.wait_for_everyone()

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """Build model framework for full fine-tuning"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    
    # Build the full framework
    model = build_framework(cfg)
    
    # For full fine-tuning, all parameters are trainable by default
    # Ensure all parameters have requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def prepare_data(cfg) -> DataLoader:
    """Prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler for full fine-tuning"""
    # Build parameter groups - all parameters will be included
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


def freeze_backbones(model, freeze_modules=""):
    """Freeze specified modules if needed"""
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


class AccelerateFFTTrainer:
    """Trainer for Full Fine-Tuning VLA on Multi-GPU"""
    
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

        # Apply any freezing from config (optional, for FFT usually nothing is frozen)
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
            logger.info("No pretrained checkpoint provided. Starting FFT training from scratch.")
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
            # UDL / IterableDataset loaders may have no len(DataLoader) (infinite stream).
            try:
                dl_len = len(self.vla_train_dataloader)
                metrics["epoch"] = round(self.completed_steps / dl_len, 2) if dl_len else 0.0
            except TypeError:
                # No batches-per-epoch; report progress through max_train_steps instead.
                max_s = max(getattr(self.config.trainer, "max_train_steps", 1), 1)
                metrics["epoch"] = round(self.completed_steps / max_s, 4)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics.get('action_dit_loss', 'N/A'):.4f}, LR: {metrics['learning_rate']:.2e}")

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

        while self.completed_steps < self.config.trainer.max_train_steps:
            # Get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # Execute training step
            t_start_model = time.perf_counter()
            step_loss = self._train_step(batch_vla)
            accumulation_loss += step_loss
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
                accumulation_loss = 0.0
                
                self.completed_steps += 1
                if self.accelerator.is_local_main_process:
                    progress_bar.update(1)

                    progress_bar.set_postfix({
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "model": f"{t_end_model - t_start_model:.3f}s",
                        "loss": f"{avg_loss:.4f}"
                    })

                # Record metrics
                step_metrics = {
                    "action_dit_loss": avg_loss,
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
        logger.info("***** Full Fine-Tuning (FFT) Multi-GPU Training Configuration *****")
        logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
        logger.info(f"  Batch size = {self.config.datasets.vla_data.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.total_batch_size}")
        logger.info(f"  Device = {self.device}")
        logger.info(f"  Training mode = Full Fine-Tuning (all parameters trainable)")

    def _train_step(self, batch_vla):
        """Execute single training step"""
        # VLA task forward propagation with mixed precision (bf16)
        # Note: GradScaler is not needed for bfloat16 as it has wider dynamic range
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            # Scale loss for gradient accumulation
            loss = action_loss / self.gradient_accumulation_steps

        # Backward propagation (no scaler needed for bf16)
        self.accelerator.backward(loss)

        return action_loss.item()

    def _finalize_training(self):
        """Training end processing"""
        final_checkpoint = os.path.join(self.config.output_dir, "final_fft_model")
        if self.accelerator.is_main_process:
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final FFT model saved at {final_checkpoint}")
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Full Fine-Tuning (FFT) Multi-GPU Training :: Warming Up")
    
    # Set device
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    
    # Create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    
    # Build model (all parameters trainable for FFT)
    vla = build_model(cfg)
    
    # Prepare data
    vla_train_dataloader = prepare_data(cfg=cfg)

    # Set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # Create trainer
    trainer = AccelerateFFTTrainer(
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

    logger.info("Full Fine-Tuning finished.")


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
