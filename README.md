# VLA-GSE: Boosting Parameter Efficient Finetuning in VLA with Generalized and Specialized Experts

This repository contains the code for
**"VLA-GSE: Boosting Parameter Efficient Finetuning in VLA with Generalized and Specialized Experts"**.

VLA-GSE is a parameter-efficient fine-tuning (PEFT) framework for Vision-Language-Action
(VLA) models. Starting from a frozen pre-trained VLM backbone, it inserts into each target
linear layer

- one **always-active generalized expert**, initialized from the leading singular
  components of the backbone weight to preserve domain-general knowledge, and
- a set of **routed specialized experts**, initialized from disjoint residual singular
  components and selected per token by a top-$k$ router, to capture context-dependent
  control adaptation.

Together with expert-wise spectral scaling, initialization-alignment, and an auxiliary
load-balancing loss, VLA-GSE updates only 2.51% of the full model parameters while
outperforming full fine-tuning on both LIBERO, LIBERO-plus, and real-world manipulation.

This project builds on the open-source VLA community codebase. We sincerely
acknowledge the open-source VLA contributors for the training framework, model interfaces,
data pipeline, and deployment utilities that this release extends with VLA-GSE and
additional PEFT baselines.

```text
.
├── LIBERO/             # Standard LIBERO benchmark: train/eval scripts
├── LIBERO-plus/        # LIBERO-plus zero-shot benchmark: train/eval scripts
├── deployment/         # Policy servers used by simulation / real-robot clients
├── VLA_GSE/            # Core library: model, dataloader, GSE/GOAT PEFT, training
├── pyproject.toml      # uv-managed project metadata and dependencies
└── README.md
```

---

## 1. Environment Setup (with `uv`)

We use [`uv`](https://docs.astral.sh/uv/) to manage the Python environment. If
`uv` is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, synchronize the environment:

```bash
# Create the virtual environment (Python 3.10) and install main + dev deps.
uv sync

# For running simulation evaluation, also install the eval dependency group.
uv sync --group eval
```

After `uv sync`, activate the environment and confirm the dependencies are visible:

```bash
source .venv/bin/activate
python -c "import torch, transformers, peft, accelerate, deepspeed; print(torch.__version__)"
```

> **Note on the LIBERO simulator.** The LIBERO / LIBERO-plus simulators require an
> OpenGL-capable environment and the upstream
> [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) package. It is recommended
> to keep simulation dependencies in a **separate** conda/uv environment from the
> training environment. The training side (`VLA-GSE`) and the simulation side
> (`LIBERO`) communicate over a WebSocket policy server, as described below.

---

## 2. Data and Pretrained Weights

By default, all scripts assume the following layout (configurable via environment
variables):

```text
./playground
├── Pretrained_models/Qwen3-VL-4B-Instruct/     # VLM backbone
└── Datasets/LEROBOT_LIBERO_DATA/               # LIBERO suites in LeRobot format
```

### 2.1 Download the VLM backbone

```bash
mkdir -p ./playground/Pretrained_models
hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

### 2.2 Download the LIBERO datasets

A one-shot helper is provided:

```bash
export DEST=/path/to/your/data/directory
bash LIBERO/data_preparation.sh
```

This downloads the four LIBERO suites (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`) in LeRobot format and symlinks them to
`./playground/Datasets/LEROBOT_LIBERO_DATA`.

---

## 3. Training

VLA-GSE and all baselines share the same config and command-line interface. Each
training script exposes four environment variables for user-configurable paths:

| Variable | Default | Description |
| ---------- | ------- | ----------- |
| `REPO_ROOT` | `$(pwd)` | Path to this repository. |
| `BASE_VLM` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | VLM backbone directory. |
| `LIBERO_DATA_ROOT` | `./playground/Datasets/LEROBOT_LIBERO_DATA` | Location of LIBERO data. |
| `CONFIG_YAML` | `./LIBERO-plus/train_files/vla_gse_cotrain_libero.yaml` | Training config. |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | GPUs to use. |

### 3.1 VLA-GSE

```bash
# 8-GPU training of VLA-GSE on all four LIBERO suites
bash LIBERO-plus/train_files/run_libero_gse.sh
```

Checkpoints are written to `./results/Checkpoints/libero_gse/`.

### 3.2 Baselines

| Script | Method | Notes |
| ------ | ------ | ----- |
| `LIBERO-plus/train_files/run_libero_fft.sh` | Full Fine-Tuning (FFT) | All parameters trainable |
| `LIBERO-plus/train_files/run_libero_lora.sh` | LoRA | Multi-GPU via `accelerate` |
| `LIBERO-plus/train_files/run_libero_other_peft.sh` | Other PEFT methods | Select with `PEFT_METHOD` |
| `LIBERO-plus/train_files/run_libero_goat.sh` | GOAT (gated MoE LoRA) | Ablation baseline |
| `LIBERO-plus/train_files/fft_gse.sh` | FFT → GSE | Two-stage fine-tuning from an FFT ckpt |

The identical set of scripts exists under `LIBERO/train_files/` for the standard
LIBERO benchmark (Table 6 in the paper).

### 3.3 Other PEFT Baselines

The unified other-PEFT trainer supports method selection through `PEFT_METHOD`.
Available values are:

```bash
lora rslora dora pissa molora adamole hydralora milora
```

Examples:

```bash
# Train rsLoRA
PEFT_METHOD=rslora bash LIBERO-plus/train_files/run_libero_other_peft.sh

# Train DoRA
PEFT_METHOD=dora bash LIBERO-plus/train_files/run_libero_other_peft.sh

# Train PiSSA
PEFT_METHOD=pissa bash LIBERO-plus/train_files/run_libero_other_peft.sh
```

All training launchers default to `NUM_GPUS=8`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
per-device batch size `16`, and `gradient_accumulation_steps=1`. Shared PEFT
hyperparameters can be overridden with `PEFT_R`, `PEFT_ALPHA`, `PEFT_DROPOUT`,
`PEFT_TARGET_MODULES`, and `PEFT_BIAS`.

`LoRA`, `rsLoRA`, `DoRA`, and `PiSSA` are configured through PEFT's native
`LoraConfig` options. `MoLoRA`, `AdaMoLE`, `HydraLoRA`, and `MiLoRA` are exposed
through the same selectable baseline entry point so their runs can share the
same training/evaluation pipeline and checkpoint naming convention; if a method
is not natively available in the installed PEFT version, the script falls back to
a LoRA-compatible adapter and records the selected method in the run config.

### 3.4 Reference hyperparameters

The default values in the scripts reproduce the paper configuration
(Appendix Table, "Hyperparameters and Configuration for VLA-GSE Fine-tuning"):

```text
rank r = 16       num_experts = 8       generalized_experts = 1       top_k = 2
s_g = 2           aux_loss_weight = 0.01
batch_size = 16 per GPU (effective 128 on 8xA100)       total_steps = 80,000
lr_vlm = 1e-5     lr_action_head = 1e-4
```

Trainable params: **114.04M (2.51% of 4.55B)** — split as 48.41M in GSE modules
and 65.62M in the action head.

---

## 4. Evaluation

Evaluation is done in a two-process setup:

1. Start a **policy server** (in the `VLA-GSE` environment) that loads a
   checkpoint and serves predictions over WebSocket.
2. Run a **simulation client** (in the `LIBERO` / `LIBERO-plus` environment) that
   steps through tasks and queries the server.

### 4.1 LIBERO-plus (zero-shot generalization, main benchmark)

```bash
# Terminal 1 (VLA-GSE env): launch the GSE policy server
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO-plus/eval_files/policy_gse.sh

# Terminal 2 (LIBERO-plus env): run all four suites in parallel on ports 5696-5699
LIBERO_HOME=./playground/LIBERO-plus \
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO-plus/eval_files/eval_libero_plus.sh
```

Per-category success rates (Camera / Robot / Language / Light / Background / Noise /
Layout) are logged in `LIBERO-plus/eval_files/eval_logs/<suite>/*.log`.

### 4.2 Standard LIBERO (Appendix Table 6)

```bash
# Terminal 1 (VLA-GSE env)
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO/eval_files/policy_gse.sh

# Terminal 2 (LIBERO env)
LIBERO_HOME=./playground/LIBERO \
TASK_SUITE_NAME=libero_object \
CKPT=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt \
bash LIBERO/eval_files/eval_libero.sh
```

Valid `TASK_SUITE_NAME` values: `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`.

### 4.3 Baselines

Each baseline has a matching policy launcher:

```bash
bash LIBERO-plus/eval_files/policy_fft.sh     # FFT
bash LIBERO-plus/eval_files/policy_lora.sh    # LoRA
bash LIBERO-plus/eval_files/policy_goat.sh    # GOAT
PEFT_METHOD=rslora bash LIBERO-plus/eval_files/policy_other_peft.sh
```

They can be combined with the same `eval_libero_plus.sh` / `eval_libero.sh` driver.

---

## 5. Repository Layout

```text
VLA_GSE/
├── config/              # Default YAML configs (training / deepspeed)
├── dataloader/          # LeRobot + VLM co-training datasets
├── model/
│   ├── framework/       # QwenFM / QwenOFT VLA frameworks
│   └── modules/         # VLM, action head, projector modules
├── gse_peft/            # VLA-GSE implementation (config, layer, model)
├── goat_peft/           # GOAT MoE-LoRA baseline implementation
└── training/
    ├── train_fft.py              # Full Fine-Tuning baseline
    ├── train_lora.py             # LoRA baseline (multi-GPU)
    ├── train_other_peft.py       # Selectable LoRA-family PEFT baselines
    ├── train_goat.py             # GOAT baseline
    ├── train_gse.py              # VLA-GSE trainer
    └── fft_gse_accumulate.py     # FFT -> GSE continued training

deployment/model_server/
├── server_fft.py / server_lora.py / server_other_peft.py / server_goat.py / server_gse.py
└── server_policy.py
```

The core VLA-GSE algorithm (SVD-based expert initialization, optimization-scale
balancing, initialization alignment, auxiliary load-balancing loss) is implemented
in `VLA_GSE/gse_peft/gse/{config,layer,model}.py` and applied to the VLM backbone
by `apply_gse_to_vlm(...)` inside `VLA_GSE/training/train_gse.py`.

---

## 6. Reproducibility

This repository reproduces all tables in the paper:

- Table 1 / Table 2: `run_libero_gse.sh` trained on the combined LIBERO train set,
  evaluated with `eval_libero_plus.sh`.
- Table 3 (ablations): toggling `gse_num_generalized_experts`, `gse_init_type`,
  and `gse_aux_loss_weight` inside `run_libero_gse.sh`.
- Table 6 / Table 7 (LIBERO main): `run_libero_*.sh` under `LIBERO/train_files/`,
  evaluated with `eval_libero.sh` on each suite.

Default random seeds and batch compositions match the paper configuration. Training
takes ≈48 hours on 8 × NVIDIA A100 GPUs for 80k steps.

---

## 7. License

This code is released under the MIT License.
