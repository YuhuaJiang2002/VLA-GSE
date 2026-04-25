#!/bin/bash
###########################################################################################
# VLA-GSE training on LIBERO (8 GPUs, no gradient accumulation).
#
# Before running:
#   - Activate your python environment (created via uv, see README.md).
#   - Set the placeholder paths below to match your local environment.
###########################################################################################

# === User configurable paths (edit before running) =======================================
REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
base_vlm=${BASE_VLM:-"./playground/Pretrained_models/Qwen3-VL-4B-Instruct"}
libero_data_root=${LIBERO_DATA_ROOT:-"./playground/Datasets/LEROBOT_LIBERO_DATA"}
config_yaml=${CONFIG_YAML:-"./LIBERO-plus/train_files/vla_gse_cotrain_libero.yaml"}
accelerate_config=${ACCELERATE_CONFIG:-"./VLA_GSE/config/deepseeds/deepspeed_zero2.yaml"}
# =========================================================================================

Framework_name=QwenOFT
freeze_module_list=''
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=libero_gse

# GSE (Generalized and Specialized Expert) configuration
gse_r=16                         # Total rank across all experts
gse_alpha=32                     # Alpha scaling factor
gse_dropout=0.05                 # Dropout rate
gse_target_modules="all-linear"
gse_bias="none"
gse_num_experts=8                # Total number of experts per block
gse_num_generalized_experts=1    # Number of generalized experts (always active)
gse_top_k=2                      # Top-k specialized experts selected by the router
gse_init_type="gse"              # Initialization type
gse_init_cof=1.0                 # Initialization coefficient
gse_aux_loss_weight=0.01         # Weight for auxiliary load-balancing loss

# Training configuration
per_device_batch_size=16
learning_rate=2.5e-05
learning_rate_vlm=1.0e-05
learning_rate_action=1.0e-04
num_warmup_steps=500
epochs=5
max_train_steps=80000

NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

cd "${REPO_ROOT}"

accelerate launch \
  --config_file ${accelerate_config} \
  --num_processes ${NUM_GPUS} \
  VLA_GSE/training/train_gse.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.epochs ${epochs} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --trainer.learning_rate.base ${learning_rate} \
  --trainer.learning_rate.qwen_vl_interface ${learning_rate_vlm} \
  --trainer.learning_rate.action_model ${learning_rate_action} \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.gse.r ${gse_r} \
  --trainer.gse.lora_alpha ${gse_alpha} \
  --trainer.gse.lora_dropout ${gse_dropout} \
  --trainer.gse.target_modules ${gse_target_modules} \
  --trainer.gse.bias ${gse_bias} \
  --trainer.gse.num_experts ${gse_num_experts} \
  --trainer.gse.num_generalized_experts ${gse_num_generalized_experts} \
  --trainer.gse.top_k ${gse_top_k} \
  --trainer.gse.init_type ${gse_init_type} \
  --trainer.gse.init_cof ${gse_init_cof} \
  --trainer.gse.aux_loss_weight ${gse_aux_loss_weight} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project VLA_GSE_LIBERO \
  --wandb_entity none
