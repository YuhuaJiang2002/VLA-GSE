#!/bin/bash
###########################################################################################
# Configurable PEFT baseline on LIBERO (8 GPUs, no gradient accumulation).
#
# Supported PEFT_METHOD values:
#   lora, rslora, dora, pissa, molora, adamole, hydralora, milora
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

peft_method=${PEFT_METHOD:-lora}
run_id=${RUN_ID:-libero_${peft_method}}

# Shared PEFT configuration
peft_r=${PEFT_R:-16}
peft_alpha=${PEFT_ALPHA:-32}
peft_dropout=${PEFT_DROPOUT:-0.05}
peft_target_modules=${PEFT_TARGET_MODULES:-"all-linear"}
peft_bias=${PEFT_BIAS:-"none"}

per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-16}
learning_rate=${LEARNING_RATE:-2.5e-05}
learning_rate_vlm=${LEARNING_RATE_VLM:-1.0e-05}
learning_rate_action=${LEARNING_RATE_ACTION:-1.0e-04}
num_warmup_steps=${NUM_WARMUP_STEPS:-500}
epochs=${EPOCHS:-5}
max_train_steps=${MAX_TRAIN_STEPS:-80000}

NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

cd "${REPO_ROOT}"

accelerate launch \
  --config_file ${accelerate_config} \
  --num_processes ${NUM_GPUS} \
  VLA_GSE/training/train_other_peft.py \
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
  --trainer.other_peft.method ${peft_method} \
  --trainer.other_peft.r ${peft_r} \
  --trainer.other_peft.lora_alpha ${peft_alpha} \
  --trainer.other_peft.lora_dropout ${peft_dropout} \
  --trainer.other_peft.target_modules ${peft_target_modules} \
  --trainer.other_peft.bias ${peft_bias} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project VLA_GSE_LIBERO_OTHER_PEFT \
  --wandb_entity none
