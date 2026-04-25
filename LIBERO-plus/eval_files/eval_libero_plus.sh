#!/bin/bash
###########################################################################################
# LIBERO-plus zero-shot evaluation driver.
#
# Usage:
#   1. Launch a policy server in another shell, e.g.:
#        bash LIBERO-plus/eval_files/policy_gse.sh
#   2. Edit the paths below (REPO_ROOT, LIBERO_HOME, CKPT, BASE_PORT).
#   3. Run this script. Evaluation across the four LIBERO-plus suites is
#      parallelised on a single host, one per port in the range [BASE_PORT, BASE_PORT+3].
###########################################################################################

# === User configurable paths (edit before running) =======================================
REPO_ROOT=${REPO_ROOT:-"$(pwd)"}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
LIBERO_HOME=${LIBERO_HOME:-"./playground/LIBERO-plus"}
CKPT=${CKPT:-"./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt"}
HOST=${HOST:-"0.0.0.0"}
BASE_PORT=${BASE_PORT:-5696}
UNNORM_KEY=${UNNORM_KEY:-"franka"}
# =========================================================================================

export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}
export DEBUG=${DEBUG:-true}

cd "${REPO_ROOT}"

SAVE_VIDEO_MODE=${SAVE_VIDEO_MODE:-"not_save"}    # "save" or "not_save"
num_trials_per_task=${NUM_TRIALS_PER_TASK:-1}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
EVAL_LOG_BASE="./LIBERO-plus/eval_files/eval_logs"

SAVE_VIDEO_FLAG=""
if [ "$SAVE_VIDEO_MODE" = "not_save" ]; then
    SAVE_VIDEO_FLAG="--args.no-save-video"
fi

run_eval() {
    local suite=$1
    local port=$2
    local log_dir="${EVAL_LOG_BASE}/${suite}"
    local log_file="${log_dir}/${folder_name}_${TIMESTAMP}.log"
    local video_path="./results/${suite}/${folder_name}"
    mkdir -p "${log_dir}"

    python ./LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path ${CKPT} \
        --args.host "$HOST" \
        --args.port "$port" \
        --args.task-suite-name "$suite" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_path" \
        ${SAVE_VIDEO_FLAG} \
        > "${log_file}" 2>&1

    echo "[DONE] ${suite} (port ${port}) - log: ${log_file}"
}

BG_PIDS=()
NUM_SUITES=${#TASK_SUITES[@]}
LAST_IDX=$((NUM_SUITES - 1))

for i in $(seq 0 $((LAST_IDX - 1))); do
    suite=${TASK_SUITES[$i]}
    port=$((BASE_PORT + i))
    echo "[START] ${suite} (background, port ${port})"
    run_eval "$suite" "$port" &
    BG_PIDS+=($!)
done

LAST_SUITE=${TASK_SUITES[$LAST_IDX]}
LAST_PORT=$((BASE_PORT + LAST_IDX))
LAST_LOG_DIR="${EVAL_LOG_BASE}/${LAST_SUITE}"
LAST_LOG_FILE="${LAST_LOG_DIR}/${folder_name}_${TIMESTAMP}.log"
mkdir -p "${LAST_LOG_DIR}"

echo "[START] ${LAST_SUITE} (foreground, port ${LAST_PORT})"
echo "------------------------------------------------------"

python ./LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${CKPT} \
    --args.host "$HOST" \
    --args.port "$LAST_PORT" \
    --args.task-suite-name "$LAST_SUITE" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "./results/${LAST_SUITE}/${folder_name}" \
    ${SAVE_VIDEO_FLAG} \
    2>&1 | tee "${LAST_LOG_FILE}"

echo ""
echo "======================================================"
echo "  Waiting for background tasks to finish..."
echo "======================================================"
for pid in "${BG_PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "======================================================"
echo "  All evaluations complete. Logs saved to:"
echo "======================================================"
for suite in "${TASK_SUITES[@]}"; do
    echo "  ${suite}: ${EVAL_LOG_BASE}/${suite}/${folder_name}_${TIMESTAMP}.log"
done
echo "======================================================"
