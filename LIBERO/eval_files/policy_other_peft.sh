#!/bin/bash
###########################################################################################
# Launches a configurable PEFT policy server for LIBERO evaluation.
#
# Use PEFT_METHOD only as metadata for the run_id / checkpoint path; the server reconstructs
# the exact adapter from the saved training config.
###########################################################################################

REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
peft_method=${PEFT_METHOD:-lora}
your_ckpt=${CKPT:-"./results/Checkpoints/libero_${peft_method}/checkpoints/steps_80000_pytorch_model.pt"}
gpu_id=${GPU_ID:-0}
port=${PORT:-5696}

cd "${REPO_ROOT}"

echo "Starting configurable PEFT policy server (${peft_method})..."
CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_other_peft.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
