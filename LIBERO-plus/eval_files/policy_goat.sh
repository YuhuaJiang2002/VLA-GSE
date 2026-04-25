#!/bin/bash
###########################################################################################
# Launches the GOAT policy server for LIBERO evaluation.
###########################################################################################

REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
your_ckpt=${CKPT:-"./results/Checkpoints/libero_goat/checkpoints/steps_80000_pytorch_model.pt"}
gpu_id=${GPU_ID:-0}
port=${PORT:-5696}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
cd "${REPO_ROOT}"

echo "Starting GOAT policy server..."
CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_goat.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
