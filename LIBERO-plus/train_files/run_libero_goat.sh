#!/bin/bash
###########################################################################################
# GOAT (gated MoE-LoRA) baseline on LIBERO (8 GPUs, no gradient accumulation).
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
run_id=libero_goat

# GOAT configuration
goat_r=16
goat_alpha=32
goat_dropout=0.05
goat_target_modules="all-linear"
goat_bias="none"
goat_num_experts=8
goat_top_k=2
goat_init_type="goat"
goat_init_cof=1.0
goat_aux_loss_weight=0.01

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
  VLA_GSE/training/train_goat.py \
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
  --trainer.goat.r ${goat_r} \
  --trainer.goat.lora_alpha ${goat_alpha} \
  --trainer.goat.lora_dropout ${goat_dropout} \
  --trainer.goat.target_modules ${goat_target_modules} \
  --trainer.goat.bias ${goat_bias} \
  --trainer.goat.num_experts ${goat_num_experts} \
  --trainer.goat.top_k ${goat_top_k} \
  --trainer.goat.init_type ${goat_init_type} \
  --trainer.goat.init_cof ${goat_init_cof} \
  --trainer.goat.aux_loss_weight ${goat_aux_loss_weight} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project VLA_GSE_LIBERO_GOAT \
  --wandb_entity none
