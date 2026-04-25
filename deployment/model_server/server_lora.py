# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 

"""
Server for LoRA-finetuned VLA models.
Loads models trained with train_lora_gpu1.py and serves them via WebSocket.
"""

import logging
import socket
import argparse
import os
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, TaskType

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from VLA_GSE.model.framework import build_framework


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def apply_lora_to_vlm(model, cfg):
    """
    Apply LoRA to the VLM backbone (same as in training).
    
    Args:
        model: The full VLA model
        cfg: Configuration object containing LoRA settings
    
    Returns:
        model: Model with LoRA applied to VLM backbone
    """
    # Get LoRA config from yaml or use defaults
    lora_cfg = cfg.trainer.get("lora", {})
    if not lora_cfg:
        lora_cfg = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05,
            "bias": "none",
        }
        logger.info("Using default LoRA configuration")
    
    logger.info(f"Applying LoRA with config: r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('lora_alpha', 32)}")
    
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
        logger.warning("Could not find VLM model to apply LoRA.")
        return model
    
    logger.info(f"Found VLM model at path: {vlm_attr_path}")
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply PEFT to the VLM model
    peft_vlm_model = get_peft_model(vlm_model, peft_config)
    
    # Replace the original VLM model with the PEFT model
    if vlm_attr_path == "qwen_vl_interface.model":
        model.qwen_vl_interface.model = peft_vlm_model
    elif vlm_attr_path == "vlm":
        model.vlm = peft_vlm_model
    elif vlm_attr_path == "qwen_vl.model":
        model.qwen_vl.model = peft_vlm_model
    
    return model


def load_lora_model(ckpt_path: str):
    """
    Load a LoRA-finetuned VLA model from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file (e.g., steps_1000_pytorch_model.pt)
    
    Returns:
        model: Loaded model with LoRA weights
        norm_stats: Dataset normalization statistics
    """
    checkpoint_pt = Path(ckpt_path)
    
    # Validate checkpoint path
    if not checkpoint_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    assert checkpoint_pt.suffix == ".pt", f"Expected .pt file, got {checkpoint_pt.suffix}"
    
    # Get run directory (checkpoint is in <run_dir>/checkpoints/<name>.pt)
    run_dir = checkpoint_pt.parents[1]
    
    # Load config
    config_yaml = run_dir / "config.yaml"
    if not config_yaml.exists():
        # Try config.json as fallback
        config_json = run_dir / "config.json"
        if config_json.exists():
            with open(config_json, "r") as f:
                cfg_dict = json.load(f)
            cfg = OmegaConf.create(cfg_dict)
        else:
            raise FileNotFoundError(f"Missing config file in {run_dir}")
    else:
        cfg = OmegaConf.load(str(config_yaml))
    
    logger.info(f"✅ Loaded config from {run_dir}")
    
    # Load dataset statistics for action denormalization
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if dataset_statistics_json.exists():
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
        logger.info(f"✅ Loaded dataset statistics from {dataset_statistics_json}")
    else:
        logger.warning(f"⚠️ No dataset_statistics.json found, action unnormalization may not work")
        norm_stats = {}
    
    # Build model from config
    logger.info(f"Building model with framework: {cfg.framework.name}")
    model = build_framework(cfg)
    
    # Apply LoRA to VLM (must match training configuration)
    model = apply_lora_to_vlm(model, cfg)
    
    # Load checkpoint weights
    logger.info(f"📦 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Load state dict with strict=False to handle potential key mismatches
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    
    # Check for key differences
    common_keys = model_keys.intersection(checkpoint_keys)
    missing_keys = model_keys - common_keys
    unexpected_keys = checkpoint_keys - common_keys
    
    if missing_keys:
        logger.warning(f"⚠️ Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}...")
    if unexpected_keys:
        logger.warning(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}...")
    
    # Load weights
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"✅ Loaded {len(common_keys)} matching keys from checkpoint")
    
    # Attach norm_stats for action unnormalization
    model.norm_stats = norm_stats
    
    return model


def main(args) -> None:
    """Main entry point for LoRA model server."""
    
    # Load LoRA model
    logger.info(f"Loading LoRA model from: {args.ckpt_path}")
    vla = load_lora_model(args.ckpt_path)
    
    # Move to device and set precision
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
        logger.info("Using bfloat16 precision")
    
    vla = vla.to("cuda").eval()
    logger.info("✅ Model loaded and moved to CUDA")
    
    # Get network info
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(f"Creating server (host: {hostname}, ip: {local_ip})")
    
    # Start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env", "model_type": "lora"},
    )
    logger.info(f"🚀 Server running on port {args.port}...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="LoRA VLA Model Server")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True,
        help="Path to LoRA checkpoint (e.g., results/Checkpoints/lora/checkpoints/steps_1000_pytorch_model.pt)"
    )
    parser.add_argument("--port", type=int, default=10093, help="Server port")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument(
        "--idle_timeout", 
        type=int, 
        default=-1, 
        help="Idle timeout in seconds, -1 means never close"
    )
    return parser

def start_debugpy_once():
    """Start debugpy once for debugging."""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    
    # if os.getenv("DEBUG", False):
    #     print("🔍 DEBUGPY is enabled")
    #     start_debugpy_once()
    
    main(args)
