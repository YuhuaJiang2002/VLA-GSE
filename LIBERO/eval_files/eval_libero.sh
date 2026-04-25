#!/bin/bash
###########################################################################################
# Standard LIBERO evaluation driver (single task suite).
#
# Usage:
#   1. Launch a policy server in another shell, e.g.:
#        bash LIBERO/eval_files/policy_gse.sh
#   2. Edit the configurable paths below.
#   3. Run this script.
###########################################################################################

# === User configurable paths (edit before running) =======================================
REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
LIBERO_HOME=${LIBERO_HOME:-"./playground/LIBERO"}
CKPT=${CKPT:-"./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-5696}
UNNORM_KEY=${UNNORM_KEY:-"franka"}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-"libero_object"}    # libero_spatial | libero_object | libero_goal | libero_10
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
SAVE_VIDEO_MODE=${SAVE_VIDEO_MODE:-"not_save"}         # "save" or "not_save"
# =========================================================================================

export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}
export DEBUG=${DEBUG:-true}

cd "${REPO_ROOT}"

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
video_out_path="./results/${TASK_SUITE_NAME}/${folder_name}"

EVAL_LOG_DIR="./LIBERO/eval_files/eval_logs"
mkdir -p "${EVAL_LOG_DIR}"
LOG_FILE="${EVAL_LOG_DIR}/${TASK_SUITE_NAME}_${folder_name}_$(date +"%Y%m%d_%H%M%S").log"

SAVE_VIDEO_FLAG=""
if [ "$SAVE_VIDEO_MODE" = "not_save" ]; then
    SAVE_VIDEO_FLAG="--args.no-save-video"
fi

python ./LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${CKPT} \
    --args.host "$HOST" \
    --args.port "$PORT" \
    --args.task-suite-name "$TASK_SUITE_NAME" \
    --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
    --args.video-out-path "$video_out_path" \
    ${SAVE_VIDEO_FLAG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Evaluation log saved to: ${LOG_FILE}"
