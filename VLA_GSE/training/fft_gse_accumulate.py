# Copyright 2025 VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");


"""
FFT-to-GSE trainer: Load a fully fine-tuned VLA checkpoint, apply GSE (Generalized
and Specialized Expert MoE LoRA) decomposition via SVD, then continue training.

Based on train_gse_accumulate.py, with the following key addition:
  - trainer.base_vla: path to a *full* VLA checkpoint (no GSE structure).
    The checkpoint is loaded BEFORE GSE is applied, so that SVD decomposes
    the pretrained weight matrices into expert initializations.

Workflow (FFT-to-GSE):
  1. build_framework() creates the model architecture from base_vlm
  2. Load base_vla checkpoint (full VLA weights, no GSE keys)
  3. apply_gse_to_vlm() decomposes loaded weights via SVD into GSE experts
  4. Freeze non-GSE VLM params; train GSE adapters + action_model
  5. Training starts from step 0

Also supports standard GSE training (no base_vla) and checkpoint resume.
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
    """Create output directory and save config."""
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
        skip_svd: If True, skip SVD computation during init (model structure and
                   scaling are still computed correctly from the original init_type).
                   Use when a checkpoint will overwrite all weights anyway.
        gse_cache_path: If provided, the post-SVD model state_dict is cached to
                   this path on the first run and loaded directly on subsequent
                   runs, eliminating redundant SVD computations.

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
                f"init_type={init_type}, skip_svd={skip_svd}, "
                f"cache={'exists' if (gse_cache_path and os.path.isfile(gse_cache_path)) else gse_cache_path or 'none'}")

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


def build_model(cfg, skip_svd=False, base_vla=None, gse_cache_path=None) -> torch.nn.Module:
    """
    Build model framework with GSE applied to VLM.

    Two modes:
      1. FFT-to-GSE (base_vla provided):
         Build architecture -> load full VLA checkpoint -> apply GSE with SVD.
         skip_svd is forced to False because SVD must decompose the loaded weights.

      2. Standard GSE (base_vla=None):
         Build architecture -> apply GSE (optionally skip SVD for checkpoint resume).

    Args:
        cfg: Configuration object
        skip_svd: If True, skip expensive SVD initialization for GSE layers.
        base_vla: Path to a full VLA checkpoint (no GSE keys) to load before
                  applying GSE decomposition.
        gse_cache_path: If set, the post-SVD state_dict is cached here so that
                  subsequent runs can skip the SVD step entirely.
    """
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")

    model = build_framework(cfg)

    if base_vla:
        logger.info(f"FFT-to-GSE: Loading pretrained full VLA checkpoint: {base_vla}")
        checkpoint = torch.load(base_vla, map_location="cpu")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"VLA checkpoint loaded: {len(missing)} missing keys, "
                    f"{len(unexpected)} unexpected keys")
        if missing:
            for k in missing[:5]:
                logger.debug(f"  missing: {k}")
        if unexpected:
            for k in unexpected[:5]:
                logger.debug(f"  unexpected: {k}")
        del checkpoint
        torch.cuda.empty_cache()

        model = apply_gse_to_vlm(model, cfg, skip_svd=False, gse_cache_path=gse_cache_path)
    else:
        model = apply_gse_to_vlm(model, cfg, skip_svd=skip_svd, gse_cache_path=gse_cache_path)

    model = ensure_action_model_trainable(model)

    return model


def prepare_data(cfg) -> DataLoader:
    """Prepare training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler for GSE training."""
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
    """Print the total number of parameters and trainable parameters."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params / 1e6:.3f}M Total, {num_trainable_params / 1e6:.3f}M Trainable")
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

        logger.info(f"VLM ({vlm_path}): Frozen {frozen_count / 1e6:.3f}M non-GSE params, "
                    f"Trainable {trainable_count / 1e6:.3f}M GSE params")
    else:
        logger.warning("Could not find VLM model to freeze non-GSE parameters")

    return model


def freeze_backbones(model, freeze_modules=""):
    """Freeze specified modules (legacy function, kept for compatibility)."""
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
                logger.warning(f"Module path does not exist, cannot freeze: {path}")
                continue
    if frozen:
        logger.info(f"Frozen modules: {frozen}")
    return model


class GradAccumGSETrainer:
    """
    Trainer for GSE-based VLA finetuning on Single GPU with Gradient Accumulation.

    Uses gradient accumulation to simulate large batch sizes:
      effective_batch_size = per_device_batch_size * gradient_accumulation_steps

    Supports full checkpoint resume: model + optimizer + scheduler + completed_steps.
    """

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, device):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.completed_steps = 0

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(cfg.trainer, "gradient_accumulation_steps", 1)

        # Effective batch size = per_device_batch_size * gradient_accumulation_steps
        self.per_device_batch_size = cfg.datasets.vla_data.per_device_batch_size
        self.effective_batch_size = self.per_device_batch_size * self.gradient_accumulation_steps

        # GSE aux loss weight (only for specialized experts)
        self.aux_loss_weight = getattr(cfg.trainer.gse, "aux_loss_weight", 0.01) if hasattr(cfg.trainer, "gse") else 0.01

        # Epoch-based vs step-based training mode
        # If epochs > 0, train for the specified number of epochs (ignoring max_train_steps)
        # If epochs <= 0, fall back to max_train_steps as before
        self.max_epochs = int(getattr(cfg.trainer, "epochs", 0))
        self.use_epoch_mode = self.max_epochs > 0

    def prepare_training(self):
        """Prepare training."""
        # Set seed
        seed = getattr(self.config, "seed", 3047)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize checkpointing (loads model/optimizer/scheduler state if resuming)
        self._init_checkpointing()

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
        logger.info(f"Model moved to {self.device}")

    def _init_checkpointing(self):
        """
        Initialize checkpoint directory and handle checkpoint loading.

        Supports three modes:
          1. is_resume=True: auto-detect latest checkpoint in output_dir, load full state
          2. pretrained_checkpoint is set: load GSE model weights, resume from that step
          3. base_vla mode (neither set): model already loaded in build_model(), start from step 0
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # Mode 1: Auto-resume from latest full checkpoint (model + optimizer + scheduler)
        if is_resume:
            resume_path, resume_steps = self._get_latest_full_checkpoint(self.checkpoint_dir)
            if resume_path:
                self._load_full_checkpoint(resume_path)
                self.completed_steps = resume_steps
                logger.info(f"Full resume from checkpoint: {resume_path}, step: {self.completed_steps}")
                return
            else:
                logger.warning(f"No full checkpoint found in {self.checkpoint_dir} for resume.")
                # Fallthrough: try pretrained_checkpoint as fallback

        # Mode 2: Load model weights from pretrained checkpoint to continue training
        if pretrained_checkpoint:
            self._load_model_checkpoint(pretrained_checkpoint)
            # Parse step number from filename
            try:
                self.completed_steps = int(
                    re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1)
                )
            except (AttributeError, ValueError):
                logger.warning(f"Could not parse steps from checkpoint: {pretrained_checkpoint}, starting from step 0")
                self.completed_steps = 0

            # Fast-forward LR scheduler to match the resumed step
            if self.completed_steps > 0:
                logger.info(f"Fast-forwarding LR scheduler to step {self.completed_steps}...")
                for _ in range(self.completed_steps):
                    self.lr_scheduler.step()
                logger.info(f"LR scheduler at step {self.completed_steps}, "
                            f"current LR: {self.lr_scheduler.get_last_lr()}")

            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, "
                        f"resuming from step {self.completed_steps}")
        else:
            logger.info("No checkpoint provided. Starting GSE training from scratch.")
            self.completed_steps = 0

    def _get_latest_full_checkpoint(self, checkpoint_dir):
        """
        Find the latest full checkpoint (contains model + optimizer + scheduler).
        Full checkpoints are saved as directories: steps_{N}_full/
        """
        if not os.path.exists(checkpoint_dir):
            return None, 0

        # Look for full checkpoint directories
        full_ckpts = []
        for entry in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(full_path) and entry.startswith("steps_") and entry.endswith("_full"):
                match = re.search(r"steps_(\d+)_full", entry)
                if match:
                    step = int(match.group(1))
                    full_ckpts.append((step, full_path))

        if not full_ckpts:
            # Fallback: look for model-only .pt checkpoints
            for f in os.listdir(checkpoint_dir):
                if re.match(r"steps_(\d+)_pytorch_model\.pt$", f):
                    match = re.search(r"steps_(\d+)_pytorch_model\.pt", f)
                    if match:
                        step = int(match.group(1))
                        full_ckpts.append((step, os.path.join(checkpoint_dir, f)))

            if not full_ckpts:
                return None, 0

            latest_step, latest_path = max(full_ckpts, key=lambda x: x[0])
            # This is a model-only checkpoint, not a full one
            # Load it as a pretrained checkpoint
            self._load_model_checkpoint(latest_path)
            return None, latest_step  # Return None path to skip full load

        latest_step, latest_path = max(full_ckpts, key=lambda x: x[0])
        return latest_path, latest_step

    def _load_model_checkpoint(self, checkpoint_path):
        """Load model weights only."""
        logger.info(f"Loading model checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=False)
        logger.info("Model checkpoint loaded")

    def _load_full_checkpoint(self, checkpoint_dir):
        """Load full checkpoint: model + optimizer + scheduler + metadata."""
        logger.info(f"Loading full checkpoint from: {checkpoint_dir}")

        # Load model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(model_state, strict=False)
            logger.info("Model state loaded")

        # Load optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
            # Move optimizer state tensors to the same device as model parameters
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            logger.info("Optimizer state loaded")

        # Load scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state)
            logger.info("Scheduler state loaded")

        # Load metadata
        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.completed_steps = meta.get("completed_steps", 0)
            logger.info(f"Metadata loaded: completed_steps={self.completed_steps}")

        logger.info("Full checkpoint loaded")

    def _save_checkpoint(self):
        """
        Save full training state: model + optimizer + scheduler + metadata.
        Also saves a standalone model .pt file for compatibility with the loading logic.
        """
        step_str = f"steps_{self.completed_steps}"

        # === Save full checkpoint (directory) ===
        full_ckpt_dir = os.path.join(self.checkpoint_dir, f"{step_str}_full")
        os.makedirs(full_ckpt_dir, exist_ok=True)

        # Model state
        torch.save(self.model.state_dict(), os.path.join(full_ckpt_dir, "model.pt"))
        # Optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(full_ckpt_dir, "optimizer.pt"))
        # Scheduler state
        torch.save(self.lr_scheduler.state_dict(), os.path.join(full_ckpt_dir, "scheduler.pt"))
        # Metadata
        meta = {
            "completed_steps": self.completed_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
        }
        with open(os.path.join(full_ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # === Also save standalone model .pt for compatibility ===
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoint_dir, f"{step_str}_pytorch_model.pt"),
        )

        # Save training summary
        summary_data = {"steps": self.completed_steps}
        with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary_data) + "\n")

        # Save accessed configuration
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(
                Path(self.config.output_dir) / "config.yaml", use_original_values=False
            )

        logger.info(f"Checkpoint saved at step {self.completed_steps} "
                    f"(full: {full_ckpt_dir}, model: {step_str}_pytorch_model.pt)")

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = self.vla_epoch_count
            logger.info(
                f"Step {self.completed_steps} | "
                f"Epoch {self.vla_epoch_count} | "
                f"Loss: {metrics.get('action_loss', 'N/A'):.4f} | "
                f"Aux: {metrics.get('aux_loss', 0):.4f} | "
                f"LR: {metrics['learning_rate']:.2e} | "
                f"Effective BS: {self.effective_batch_size}"
            )

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_epoch_count = 0

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            logger.info(f"Epoch {self.vla_epoch_count} completed, starting new epoch...")
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)
        return batch_vla

    def _should_stop(self):
        """Check if training should stop based on epoch or step mode."""
        if self.use_epoch_mode:
            return self.vla_epoch_count >= self.max_epochs
        else:
            return self.completed_steps >= self.config.trainer.max_train_steps

    def train(self):
        """
        Execute training loop with proper gradient accumulation.

        Supports two termination modes:
          - Epoch mode (epochs > 0): train for the specified number of epochs
          - Step mode (epochs <= 0): train for max_train_steps optimizer steps

        Uses a separate micro_step counter to track accumulation progress,
        fixing the bug in the original train_gse.py where completed_steps
        was incorrectly used as the micro-step counter.
        """
        self._log_training_config()
        self._create_data_iterators()

        self.model.train()

        # Set up progress bar based on training mode
        if self.use_epoch_mode:
            # Estimate total optimizer steps from dataloader length and epochs
            dataloader_len = len(self.vla_train_dataloader) if hasattr(self.vla_train_dataloader, "__len__") else 0
            if dataloader_len > 0:
                estimated_total_steps = (dataloader_len * self.max_epochs) // self.gradient_accumulation_steps
            else:
                estimated_total_steps = 0
            progress_bar = tqdm(
                initial=self.completed_steps,
                total=estimated_total_steps if estimated_total_steps > 0 else None,
                desc=f"Training ({self.max_epochs} epochs)",
            )
        else:
            progress_bar = tqdm(
                range(self.completed_steps, self.config.trainer.max_train_steps),
                initial=self.completed_steps,
                total=self.config.trainer.max_train_steps,
                desc="Training (step mode)",
            )

        micro_step = 0          # Tracks micro-steps within each accumulation window
        accumulation_loss = 0.0
        accumulation_aux_loss = 0.0

        while not self._should_stop():
            # Get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # In epoch mode, _get_next_batch may have incremented vla_epoch_count
            # Check if we reached the target epoch before running forward pass
            if self.use_epoch_mode and self.vla_epoch_count >= self.max_epochs:
                logger.info(f"Reached target epoch {self.max_epochs}, stopping training.")
                break

            # Execute one micro-step (forward + backward, loss scaled by 1/grad_accum)
            t_start_model = time.perf_counter()
            step_loss, aux_loss = self._train_step(batch_vla)
            accumulation_loss += step_loss
            accumulation_aux_loss += aux_loss
            t_end_model = time.perf_counter()

            micro_step += 1

            # Optimizer step after accumulating enough micro-steps
            if micro_step % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.trainer.gradient_clipping
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                # Compute average loss over the accumulation window
                avg_loss = accumulation_loss / self.gradient_accumulation_steps
                avg_aux_loss = accumulation_aux_loss / self.gradient_accumulation_steps
                accumulation_loss = 0.0
                accumulation_aux_loss = 0.0

                self.completed_steps += 1
                progress_bar.update(1)

                epoch_str = f"{self.vla_epoch_count}/{self.max_epochs}" if self.use_epoch_mode else str(self.vla_epoch_count)
                progress_bar.set_postfix({
                    "epoch": epoch_str,
                    "data": f"{t_end_data - t_start_data:.3f}s",
                    "model": f"{t_end_model - t_start_model:.3f}s",
                    "loss": f"{avg_loss:.4f}",
                    "aux": f"{avg_aux_loss:.4f}",
                })

                # Record metrics
                step_metrics = {
                    "action_loss": avg_loss,
                    "aux_loss": avg_aux_loss,
                    "data_time": t_end_data - t_start_data,
                    "model_time": t_end_model - t_start_model,
                }

                # Evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0 and self.completed_steps > 0:
                    step_metrics = self.eval_action_model(step_metrics)

                self._log_metrics(step_metrics)

                # Save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            # Check step-based termination
            if not self.use_epoch_mode and self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Evaluate the model on a batch."""
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
        """Record training config."""
        base_vla = getattr(self.config.trainer, "base_vla", None)
        mode_str = "FFT-to-GSE" if (base_vla and str(base_vla)) else "Standard GSE"
        logger.info("=" * 80)
        logger.info(f"{mode_str} Single GPU Training with Gradient Accumulation")
        logger.info("=" * 80)
        if self.use_epoch_mode:
            logger.info(f"  Training mode                = EPOCH (train for {self.max_epochs} epochs)")
            logger.info(f"  Max epochs                   = {self.max_epochs}")
        else:
            logger.info(f"  Training mode                = STEP (train for {self.config.trainer.max_train_steps} steps)")
            logger.info(f"  Max optimization steps       = {self.config.trainer.max_train_steps}")
        logger.info(f"  Per-device batch size        = {self.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps  = {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size          = {self.effective_batch_size}")
        logger.info(f"  Device                       = {self.device}")
        logger.info(f"  Resume from step             = {self.completed_steps}")

        # Log GSE config if available
        gse_cfg = self.config.trainer.get("gse", {})
        if gse_cfg:
            logger.info(f"  GSE rank                     = {gse_cfg.get('r', 16)}")
            logger.info(f"  GSE num_experts              = {gse_cfg.get('num_experts', 8)}")
            logger.info(f"  GSE num_generalized_experts  = {gse_cfg.get('num_generalized_experts', 2)}")
            logger.info(f"  GSE top_k                    = {gse_cfg.get('top_k', 2)}")
            logger.info(f"  GSE init_type                = {gse_cfg.get('init_type', 'gse')}")
            logger.info(f"  GSE aux_loss_weight          = {self.aux_loss_weight}")
        logger.info("=" * 80)

    def _train_step(self, batch_vla):
        """Execute single micro-step: forward + backward with loss scaled by 1/grad_accum."""
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]

            if self.aux_loss_weight > 0:
                aux_loss = torch.tensor(0.0, device=self.device)
                try:
                    vlm_model = None
                    if hasattr(self.model, "qwen_vl_interface") and hasattr(self.model.qwen_vl_interface, "model"):
                        vlm_model = self.model.qwen_vl_interface.model
                    elif hasattr(self.model, "vlm"):
                        vlm_model = self.model.vlm

                    if vlm_model is not None and hasattr(vlm_model, "get_aux_loss"):
                        aux_loss = vlm_model.get_aux_loss()
                except Exception:
                    pass

                total_loss = action_loss + self.aux_loss_weight * aux_loss
            else:
                aux_loss = 0.0
                total_loss = action_loss

            loss = total_loss / self.gradient_accumulation_steps

        loss.backward()

        return action_loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

    def _finalize_training(self):
        """Training end processing."""
        final_checkpoint = os.path.join(self.config.output_dir, "final_gse_model")
        os.makedirs(final_checkpoint, exist_ok=True)

        # Save full model state
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))

        logger.info(f"Training complete. Final GSE model saved at {final_checkpoint}")


def main(cfg) -> None:
    logger.info("VLA FFT-to-GSE Training with Gradient Accumulation :: Warming Up")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = wrap_config(cfg)
    logger.info("Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)

    # --- Determine training mode ---
    _raw = getattr(cfg.trainer, "base_vla", None)
    base_vla = _raw if (_raw and isinstance(_raw, str) and len(_raw) > 0) else None

    _raw = getattr(cfg.trainer, "pretrained_checkpoint", None)
    pretrained_checkpoint = _raw if (_raw and isinstance(_raw, str) and len(_raw) > 0) else None

    is_resume = getattr(cfg.trainer, "is_resume", False)

    # When resuming, check if output_dir already has GSE checkpoints.
    # If so, we can skip the expensive SVD (the checkpoint will overwrite weights).
    resume_ckpt_exists = False
    if is_resume:
        ckpt_dir = os.path.join(str(output_dir), "checkpoints")
        if os.path.exists(ckpt_dir):
            resume_ckpt_exists = any(
                e.startswith("steps_") and (e.endswith("_full") or e.endswith("_pytorch_model.pt"))
                for e in os.listdir(ckpt_dir)
            )

    _raw = getattr(cfg.trainer, "gse_cache_path", None)
    gse_cache_path = _raw if (_raw and isinstance(_raw, str) and len(_raw) > 0) else None
    if gse_cache_path:
        logger.info(f"GSE cache path: {gse_cache_path}")

    if base_vla and not resume_ckpt_exists:
        logger.info(f"FFT-to-GSE mode: will load full VLA from {base_vla}")
        vla = build_model(cfg, skip_svd=False, base_vla=base_vla, gse_cache_path=gse_cache_path)
    elif resume_ckpt_exists:
        logger.info("Resume mode: existing GSE checkpoint found, skipping SVD.")
        vla = build_model(cfg, skip_svd=True, base_vla=None)
    elif pretrained_checkpoint:
        logger.info(f"GSE checkpoint resume: {pretrained_checkpoint}")
        vla = build_model(cfg, skip_svd=True)
    else:
        logger.info("No base_vla or pretrained_checkpoint. Training GSE from scratch.")
        vla = build_model(cfg, skip_svd=False, gse_cache_path=gse_cache_path)

    # Prepare data
    vla_train_dataloader = prepare_data(cfg=cfg)

    # In epoch mode (epochs > 0), estimate max_train_steps from dataloader length
    # so the LR scheduler gets the correct num_training_steps
    max_epochs = int(getattr(cfg.trainer, "epochs", 0))
    grad_accum = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    if max_epochs > 0 and hasattr(vla_train_dataloader, "__len__") and len(vla_train_dataloader) > 0:
        estimated_max_steps = (len(vla_train_dataloader) * max_epochs) // grad_accum
        logger.info(f"Epoch mode: {max_epochs} epochs x {len(vla_train_dataloader)} batches / "
                    f"{grad_accum} grad_accum = ~{estimated_max_steps} optimizer steps")
        # Override max_train_steps so LR scheduler uses the correct total
        cfg.trainer.max_train_steps = estimated_max_steps

    # Set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # Create trainer
    trainer = GradAccumGSETrainer(
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

    logger.info("FFT-to-GSE gradient accumulation training finished.")


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
        print("Waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
