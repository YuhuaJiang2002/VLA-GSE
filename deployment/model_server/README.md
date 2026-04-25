# Policy Servers

Each finetuning method has a dedicated server that loads a checkpoint and exposes
the policy over a WebSocket for the LIBERO / real-robot clients.

| Server | Loads |
|--------|-------|
| `server_fft.py` | Full Fine-Tuning (FFT) checkpoints |
| `server_lora.py` | LoRA-finetuned checkpoints |
| `server_other_peft.py` | Selectable PEFT checkpoints (`lora`, `rslora`, `dora`, `pissa`, `molora`, `adamole`, `hydralora`, `milora`) |
| `server_goat.py` | GOAT (gated MoE-LoRA) checkpoints |
| `server_gse.py` | VLA-GSE checkpoints |
| `server_policy.py` | Generic pretrained checkpoints |

Minimal launch example (VLA-GSE):

```bash
your_ckpt=./results/Checkpoints/libero_gse/checkpoints/steps_80000_pytorch_model.pt

CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_gse.py \
    --ckpt_path ${your_ckpt} \
    --port 5696 \
    --use_bf16 \
    --base_vlm ./playground/Pretrained_models/Qwen3-VL-4B-Instruct \
    --skip_svd
```

Convenience launchers with default paths are provided under
`LIBERO/eval_files/policy_*.sh` and `LIBERO-plus/eval_files/policy_*.sh`.
