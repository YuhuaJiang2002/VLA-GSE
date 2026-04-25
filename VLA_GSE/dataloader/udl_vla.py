"""
Bridge UnifiedDataLoader (UDL) samples to StarVLA VLA batch format (QwenOFT / LeRobot-style).

Matches loading logic in ``udl/test.py``: OmegaConf YAML → merge ``dataset.common`` and
``dataset.train`` → ``load_dataset(**train_ds_config)``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

# Standard logging: accelerate's get_logger() requires PartialState/Accelerator before use,
# but UDL loading can run before training initializes Accelerate.
logger = logging.getLogger(__name__)


def _tensor_to_pil_rgb(img_t: torch.Tensor) -> Image.Image:
    """Convert CHW float [0,1] or uint8 tensor to PIL RGB."""
    x = img_t.detach().cpu().float()
    if x.dim() == 3 and x.shape[0] in (1, 3, 4):
        x = x.float()
        if x.max() <= 1.0 + 1e-3:
            x = (x.clamp(0, 1) * 255.0).byte()
        else:
            x = x.clamp(0, 255).byte()
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 4:
            x = x[:3]
        arr = x.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected image tensor shape {tuple(x.shape)}")
    return Image.fromarray(arr, mode="RGB")


def _collect_image_tensors(obs: dict) -> list[torch.Tensor]:
    """Ordered list of camera views (CHW), excluding masks/metadata."""
    skip = {"camera_mask", "pad_mask", "timestep"}
    keys = sorted(k for k in obs if k.startswith("image_") and k not in skip)
    if not keys and "image_primary" in obs:
        keys = ["image_primary"]
    tensors = []
    for k in keys:
        t = obs[k]
        if t.dim() == 4:
            t = t[-1]
        elif t.dim() == 5:
            t = t[0, -1]
        elif t.dim() != 3:
            raise ValueError(f"Unexpected {k} shape {tuple(t.shape)}")
        tensors.append(t)
    if not tensors:
        raise ValueError("No image_* keys in UDL observation")
    return tensors


def udl_sample_to_example(
    sample: dict[str, Any],
    action_dim: int,
    chunk_len: int,
    duplicate_single_view: bool = True,
) -> dict[str, Any]:
    """
    One UDL ``sample()`` → one StarVLA example dict:
    ``image`` (list[PIL]), ``lang`` (str), ``action`` (np.float32 [T, action_dim]).
    """
    obs = sample["observation"]
    pil_list = [_tensor_to_pil_rgb(t) for t in _collect_image_tensors(obs)]
    if duplicate_single_view and len(pil_list) == 1:
        pil_list = [pil_list[0], pil_list[0]]

    lang = sample.get("instruction", "")
    if isinstance(lang, list):
        lang = lang[0] if lang else ""
    lang = str(lang)

    if "action" not in sample or sample["action"] is None:
        raise KeyError("UDL sample missing action; set dataset.common.load_data: true in UDL YAML")

    act = sample["action"].detach().cpu().float()
    while act.dim() > 2:
        act = act.reshape(-1, act.shape[-1])
    if act.dim() == 1:
        act = act.unsqueeze(0)
    arr = act.numpy().astype(np.float32)

    d = arr.shape[-1]
    if d < action_dim:
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], action_dim - d), dtype=np.float32)], axis=-1
        )
    elif d > action_dim:
        arr = arr[:, :action_dim]

    if arr.shape[0] < chunk_len:
        pad = chunk_len - arr.shape[0]
        tail = np.tile(arr[-1:], (pad, 1))
        arr = np.concatenate([arr, tail], axis=0)

    return {"image": pil_list, "lang": lang, "action": arr}


def _pad_action_rows(batch: list[dict], action_dim: int) -> None:
    """In-place: pad ``action`` rows so ``np.array`` is rectangular [B, T, D]."""
    max_t = max(int(ex["action"].shape[0]) for ex in batch)
    for ex in batch:
        a = ex["action"]
        if a.shape[0] < max_t:
            pad = max_t - a.shape[0]
            tail = np.tile(a[-1:], (pad, 1)) if a.shape[0] else np.zeros((pad, action_dim), np.float32)
            ex["action"] = np.concatenate([a, tail], axis=0).astype(np.float32)


class UDLVLABatchIterable(IterableDataset):
    """Infinite iterable of batches (list[dict]) for StarVLA training."""

    def __init__(
        self,
        udl_ds: Any,
        batch_size: int,
        action_dim: int,
        chunk_len: int,
        max_sample_retries: int = 32,
    ):
        super().__init__()
        self.udl_ds = udl_ds
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.max_sample_retries = max_sample_retries

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            # UDL + random episode indices are not fork-safe for all backends; single worker only.
            seed = worker_info.seed + worker_info.id
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))

        while True:
            batch: list[dict] = []
            for _ in range(self.batch_size):
                last_err: Exception | None = None
                for _attempt in range(self.max_sample_retries):
                    try:
                        s = self.udl_ds.sample()
                        ex = udl_sample_to_example(s, self.action_dim, self.chunk_len)
                        batch.append(ex)
                        break
                    except Exception as e:
                        last_err = e
                else:
                    err = last_err or RuntimeError("unknown UDL sample failure")
                    raise RuntimeError(
                        f"UDL sample() failed after {self.max_sample_retries} retries"
                    ) from err

            _pad_action_rows(batch, self.action_dim)
            yield batch


class UDLTrainWrapper:
    """Holds UDL dataset for saving statistics (optional)."""

    def __init__(self, train_ds: Any):
        self.train_ds = train_ds

    def save_dataset_statistics(self, save_path: Path | str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"udl": {"source": "UnifiedDataLoader"}}

        try:
            stats = self.train_ds.stats
            serializable: dict[str, Any] = {}
            for name, st in stats.items():
                if hasattr(st, "model_dump"):
                    serializable[name] = st.model_dump(mode="json")
                elif hasattr(st, "dict"):
                    serializable[name] = st.dict()
                else:
                    serializable[name] = str(st)
            payload["udl"]["dataset_statistics"] = serializable
        except Exception as e:
            logger.warning(f"Could not export full UDL stats: {e}")

        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved UDL dataset statistics stub at {save_path}")


def load_udl_train_dataset_from_yaml(yaml_path: str | Path) -> Any:
    """
    Load the UDL **train** dataset exactly like ``udl/test.py``:

    ``OmegaConf.load`` → ``OmegaConf.to_container`` → merge ``dataset.common`` and
    ``dataset.train`` → ``load_dataset(**train_ds_config)``.
    """
    try:
        from udl import load_dataset
    except ImportError as e:
        raise ImportError(
            "Package `udl` (UnifiedDataLoader) is required. "
            "Install from your UnifiedDataLoader checkout."
        ) from e

    yaml_path = Path(yaml_path)
    config = OmegaConf.load(yaml_path)
    config = OmegaConf.to_container(config)  # same as udl/test.py
    train_ds_config = {**config["dataset"]["common"], **config["dataset"]["train"]}
    logger.info("UDL load (test.py style): %s", yaml_path)
    return load_dataset(**train_ds_config)


def build_udl_vla_dataloader_from_dataset(cfg: Any, train_ds: Any) -> DataLoader:
    """
    Wrap an already-loaded UDL train dataset (from ``load_udl_train_dataset_from_yaml``)
    in a PyTorch ``DataLoader`` that yields StarVLA-style batches.
    """
    vla = cfg.datasets.vla_data

    action_dim = int(cfg.framework.action_model.action_dim)
    past = int(cfg.framework.action_model.past_action_window_size)
    future = int(cfg.framework.action_model.future_action_window_size)
    chunk_len = past + 1 + future

    iterable = UDLVLABatchIterable(
        train_ds,
        batch_size=int(vla.per_device_batch_size),
        action_dim=action_dim,
        chunk_len=chunk_len,
    )
    wrapper = UDLTrainWrapper(train_ds)
    output_dir = Path(cfg.output_dir)
    wrapper.save_dataset_statistics(output_dir / "dataset_statistics.json")

    return DataLoader(iterable, batch_size=None, num_workers=0, pin_memory=False)


def build_udl_vla_train_dataloader(cfg: Any) -> DataLoader:
    """
    Build a DataLoader that yields batches identical in structure to LeRobot ``collate_fn`` output:
    ``list[dict]`` with ``image``, ``lang``, ``action``.
    """
    vla = cfg.datasets.vla_data
    yaml_path = getattr(vla, "udl_config_path", None) or getattr(vla, "udl_config", None)
    if not yaml_path:
        raise ValueError(
            "Set `datasets.vla_data.udl_config_path` to your UDL YAML (same as in udl/test.py)."
        )

    train_ds = load_udl_train_dataset_from_yaml(yaml_path)
    return build_udl_vla_dataloader_from_dataset(cfg, train_ds)
