# Copyright VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Server for VLA-GSE (Generalized and Specialized Expert) finetuned VLA models.
Loads models trained with train_gse.py and serves them via WebSocket.
"""

import logging
import socket
import argparse
import os
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from VLA_GSE.model.framework import build_framework
from VLA_GSE.gse_peft import GSEConfig, get_gse_model
from VLA_GSE.gse_peft.utils.peft_types import TaskType


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def apply_gse_to_vlm(model, cfg, skip_svd: bool = True):
    """
    Apply GSE to the VLM backbone for inference.

    Args:
        model: The full VLA model
        cfg: Configuration object containing GSE settings
        skip_svd: If True, skip SVD computation and load directly from checkpoint.

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
            "specialized_scaling_method": "default",
        }
        logger.info("Using default GSE configuration")

    init_type = gse_cfg.get("init_type", "gse")
    logger.info(f"Applying GSE with config: r={gse_cfg.get('r', 16)}, "
                f"num_experts={gse_cfg.get('num_experts', 8)}, "
                f"num_generalized_experts={gse_cfg.get('num_generalized_experts', 2)}, "
                f"top_k={gse_cfg.get('top_k', 2)}, "
                f"init_type={init_type}, "
                f"specialized_scaling_method={gse_cfg.get('specialized_scaling_method', 'default')}, "
                f"skip_svd={skip_svd})")

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
        logger.warning("Could not find VLM model to apply GSE.")
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
        specialized_scaling_method=gse_cfg.get("specialized_scaling_method", "default"),
        specialized_scaling_base=float(gse_cfg.get("specialized_scaling_base", 2.0)),
        specialized_scaling_eps=float(gse_cfg.get("specialized_scaling_eps", 1e-12)),
        skip_svd_init=skip_svd,
        aux_loss_weight=float(gse_cfg.get("aux_loss_weight", 0.01)),
    )

    gse_vlm_model = get_gse_model(vlm_model, gse_config)

    if vlm_attr_path == "qwen_vl_interface.model":
        model.qwen_vl_interface.model = gse_vlm_model
    elif vlm_attr_path == "vlm":
        model.vlm = gse_vlm_model
    elif vlm_attr_path == "qwen_vl.model":
        model.qwen_vl.model = gse_vlm_model

    return model


def load_gse_model(ckpt_path: str, base_vlm_override: str = None, skip_svd: bool = True):
    """
    Load a GSE-finetuned VLA model from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file (e.g., steps_1000_pytorch_model.pt)
        base_vlm_override: If set, override framework.qwenvl.base_vlm in the saved config
        skip_svd: If True, skip SVD computation and load model directly from checkpoint
    
    Returns:
        model: Loaded model with GSE weights
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

    if base_vlm_override:
        logger.info(f"Overriding base_vlm: {cfg.framework.qwenvl.base_vlm} -> {base_vlm_override}")
        cfg.framework.qwenvl.base_vlm = base_vlm_override

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
    
    # Apply GSE structure to VLM (checkpoint overwrites weights)
    model = apply_gse_to_vlm(model, cfg, skip_svd=skip_svd)
    
    # Load checkpoint weights directly by parameter name
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    common_keys = model_keys.intersection(checkpoint_keys)
    missing_keys = model_keys - common_keys
    unexpected_keys = checkpoint_keys - common_keys
    
    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}...")
    
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Loaded {len(common_keys)} matching keys from checkpoint")
    
    # Attach norm_stats for action unnormalization
    model.norm_stats = norm_stats
    
    return model


def main(args) -> None:
    """Main entry point for GSE model server."""
    
    # Load GSE model
    logger.info(f"Loading GSE model from: {args.ckpt_path}")
    vla = load_gse_model(args.ckpt_path, base_vlm_override=args.base_vlm, skip_svd=args.skip_svd)
    
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
        metadata={"env": "simpler_env", "model_type": "gse"},
    )
    logger.info(f"🚀 Server running on port {args.port}...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="GSE VLA Model Server")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True,
        help="Path to GSE checkpoint (e.g., results/Checkpoints/gse/checkpoints/steps_1000_pytorch_model.pt)"
    )
    parser.add_argument("--port", type=int, default=10093, help="Server port")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument(
        "--idle_timeout", 
        type=int, 
        default=-1, 
        help="Idle timeout in seconds, -1 means never close"
    )
    parser.add_argument(
        "--base_vlm",
        type=str,
        default=None,
        help="Override base_vlm path in config (useful when config was saved in a different environment)"
    )
    parser.add_argument(
        "--skip_svd",
        action="store_true",
        default=True,
        help="Skip SVD computation, load model directly from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-skip-svd",
        dest="skip_svd",
        action="store_false",
        help="Do not skip SVD (run SVD init before loading checkpoint)"
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
