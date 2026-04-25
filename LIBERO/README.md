# LIBERO Training and Evaluation

This document explains how to reproduce the standard **LIBERO** benchmark results
(Table 6 in the paper) with VLA-GSE. For the more challenging zero-shot evaluation
setting please refer to [`../LIBERO-plus/README.md`](../LIBERO-plus/README.md).

All commands in this document are to be executed from the repository root.

---

## 0. Data Preparation

Download the four LIBERO suites (LeRobot format) and the co-training VQA data, and
link them under `./playground/Datasets/`:

```bash
export DEST=/path/to/your/data/directory
bash LIBERO/data_preparation.sh
```

This downloads the following datasets and sets up the required symbolic links:

- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

After the script finishes, the `modality.json` metadata in
`LIBERO/train_files/modality.json` is copied into each suite's `meta/` folder.

---

## 1. Training

Five training entry points are provided under `LIBERO/train_files/`:

| Script | Method |
|--------|--------|
| `run_libero_fft.sh`  | Full fine-tuning (FFT) baseline |
| `run_libero_lora.sh` | Vanilla LoRA baseline |
| `run_libero_other_peft.sh` | Selectable PEFT baseline (`lora`, `rslora`, `dora`, `pissa`, `molora`, `adamole`, `hydralora`, `milora`) |
| `run_libero_goat.sh` | GOAT (gated MoE-LoRA) baseline |
| `run_libero_gse.sh`  | VLA-GSE (ours) |

Each script exposes four environment variables for user-configurable paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_ROOT` | `$(pwd)` | Path to this repository |
| `BASE_VLM` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | VLM backbone |
| `LIBERO_DATA_ROOT` | `./playground/Datasets/LEROBOT_LIBERO_DATA` | Downloaded datasets |
| `CONFIG_YAML` | `./LIBERO/train_files/starvla_cotrain_libero.yaml` | Training config |

To train VLA-GSE on all four LIBERO suites:

```bash
bash LIBERO/train_files/run_libero_gse.sh
```

---

## 2. Evaluation

Evaluation is split into a **policy server** (starVLA environment) and a
**simulator client** (LIBERO environment).

**Step 1 - start the policy server (VLA-GSE env):**

```bash
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO/eval_files/policy_gse.sh
```

Alternative baselines: `policy_fft.sh`, `policy_lora.sh`, `policy_other_peft.sh`,
`policy_goat.sh`.

**Step 2 - run the evaluation (LIBERO env):**

```bash
LIBERO_HOME=./playground/LIBERO \
TASK_SUITE_NAME=libero_object \
bash LIBERO/eval_files/eval_libero.sh
```

Valid `TASK_SUITE_NAME` values: `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`. Per-episode replay videos can be enabled with
`SAVE_VIDEO_MODE=save`.

Results are saved to `LIBERO/eval_files/eval_logs/<suite>/*.log`.
