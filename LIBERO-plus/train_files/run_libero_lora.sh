#!/bin/bash
###########################################################################################
# LoRA baseline on LIBERO (multi-GPU via accelerate).
###########################################################################################

# === User configurable paths (edit before running) =======================================
REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
base_vlm=${BASE_VLM:-"./playground/Pretrained_models/Qwen3-VL-4B-Instruct"}
libero_data_root=${LIBERO_DATA_ROOT:-"./playground/Datasets/LEROBOT_LIBERO_DATA"}
config_yaml=${CONFIG_YAML:-"./LIBERO-plus/train_files/starvla_cotrain_libero.yaml"}
accelerate_config=${ACCELERATE_CONFIG:-"./VLA_GSE/config/deepseeds/deepspeed_zero2.yaml"}
# =========================================================================================

Framework_name=QwenOFT
freeze_module_list=''
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=libero_lora

# LoRA configuration
lora_r=16
lora_alpha=32
lora_dropout=0.05
lora_target_modules="all-linear"
lora_bias="none"

per_device_batch_size=16
learning_rate=2.5e-05
learning_rate_vlm=1.0e-05
learning_rate_action=1.0e-04
num_warmup_steps=500
epochs=5
max_train_steps=80000

NUM_GPUS=${NUM_GPUS:-1}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

cd "${REPO_ROOT}"

accelerate launch \
  --config_file ${accelerate_config} \
  --num_processes ${NUM_GPUS} \
  VLA_GSE/training/train_lora.py \
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
  --trainer.lora.r ${lora_r} \
  --trainer.lora.lora_alpha ${lora_alpha} \
  --trainer.lora.lora_dropout ${lora_dropout} \
  --trainer.lora.target_modules ${lora_target_modules} \
  --trainer.lora.bias ${lora_bias} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project VLA_GSE_LIBERO_LoRA \
  --wandb_entity none
