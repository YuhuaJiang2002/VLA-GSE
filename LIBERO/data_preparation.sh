#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# Download the LIBERO simulation datasets (LeRobot format) used by VLA-GSE and link them
# into ./playground/Datasets/LEROBOT_LIBERO_DATA for training.
#
# Usage (from the repository root):
#   export DEST=/path/to/dir && bash LIBERO/data_preparation.sh
# or
#   bash LIBERO/data_preparation.sh /path/to/dir
###########################################################################################

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash LIBERO/data_preparation.sh"
  echo "  or: bash LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

python -m pip install -U "huggingface-hub==0.35.3"

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf download "$repo" --repo-type dataset --local-dir "$DEST/libero/${repo##*/}"
done

if [ -n "${VQA_DATASET_REPO:-}" ]; then
  hf download "$VQA_DATASET_REPO" --repo-type dataset --local-dir "$DEST/LLaVA-OneVision-COCO"
  unzip -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"
else
  echo "Skipping optional VQA co-training data. Set VQA_DATASET_REPO to download it."
fi

mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
if [ -d "$DEST/LLaVA-OneVision-COCO" ]; then
  ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"
fi

# Copy modality metadata into each suite.
cp "$CUR/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
