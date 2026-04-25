# LIBERO-plus Training and Evaluation

This document explains how to reproduce the zero-shot generalization results on the
**LIBERO-plus** benchmark (Tables 1 and 2 in the paper) with VLA-GSE. For the standard
LIBERO benchmark, please refer to [`../LIBERO/README.md`](../LIBERO/README.md).
A higher-level overview of the repository is provided in the project-level
[`../README.md`](../README.md).

All commands in this document are to be executed from the repository root.

---

## 0. Data Preparation

LIBERO-plus shares the same training corpus as the standard LIBERO benchmark
(the four LIBERO task suites in LeRobot format). Please follow the data preparation
step in [`../LIBERO/README.md`](../LIBERO/README.md) to download and link the
datasets under `./playground/Datasets/LEROBOT_LIBERO_DATA/`.

The zero-shot evaluation also requires the [LIBERO-plus](https://github.com/libero-plus/LIBERO-plus)
simulator assets; by default they are expected under `./playground/LIBERO-plus/`.

---

## 1. Training

Five training entry points are provided under `LIBERO-plus/train_files/`:

| Script | Method |
|--------|--------|
| `run_libero_fft.sh`  | Full fine-tuning (FFT) baseline |
| `run_libero_lora.sh` | Vanilla LoRA baseline |
| `run_libero_other_peft.sh` | Selectable PEFT baseline (`lora`, `rslora`, `dora`, `pissa`, `molora`, `adamole`, `hydralora`, `milora`) |
| `run_libero_goat.sh` | GOAT (gated MoE-LoRA) baseline |
| `run_libero_gse.sh`  | VLA-GSE (ours) |
| `fft_gse.sh`         | Two-stage FFT → GSE continued training |

Each script exposes four environment variables for user-configurable paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_ROOT` | `$(pwd)` | Path to this repository |
| `BASE_VLM` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | VLM backbone |
| `LIBERO_DATA_ROOT` | `./playground/Datasets/LEROBOT_LIBERO_DATA` | Downloaded datasets |
| `CONFIG_YAML` | `./LIBERO-plus/train_files/vla_gse_cotrain_libero.yaml` | Training config |

To train VLA-GSE on the combined LIBERO training set:

```bash
bash LIBERO-plus/train_files/run_libero_gse.sh
```

Checkpoints are written to `./results/Checkpoints/libero_gse/`.

---

## 2. Evaluation

Evaluation is organized as a **policy server** (VLA-GSE environment) plus a
**simulator client** (LIBERO-plus environment). The client runs all four suites in
parallel on ports `BASE_PORT` … `BASE_PORT + 3`, so the policy server must expose the
matching ports.

**Step 1 — launch the policy server (VLA-GSE env):**

```bash
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO-plus/eval_files/policy_gse.sh
```

Alternative baselines: `policy_fft.sh`, `policy_lora.sh`, `policy_other_peft.sh`,
`policy_goat.sh`.
Use `PORT=<port>` to override the listening port when launching multiple
servers on the same host.

**Step 2 — run the evaluation (LIBERO-plus env):**

```bash
LIBERO_HOME=./playground/LIBERO-plus \
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO-plus/eval_files/eval_libero_plus.sh
```

Per-suite zero-shot success rates (decomposed into Camera / Robot / Language /
Light / Background / Noise / Layout variations) are written to
`LIBERO-plus/eval_files/eval_logs/<suite>/*.log`.
