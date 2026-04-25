#!/bin/bash
###########################################################################################
# Launches the VLA-GSE policy server for LIBERO evaluation.
###########################################################################################

REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
base_vlm=${BASE_VLM:-"./playground/Pretrained_models/Qwen3-VL-4B-Instruct"}
your_ckpt=${CKPT:-"./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt"}
gpu_id=${GPU_ID:-0}
port=${PORT:-5696}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
cd "${REPO_ROOT}"

echo "Starting VLA-GSE policy server..."
CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_gse.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --base_vlm ${base_vlm} \
    --skip_svd
