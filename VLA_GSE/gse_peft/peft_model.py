"""
GSE PEFT Model wrapper
Simplified version for starVLA integration
"""
import os
import logging

import torch
from torch import nn
from transformers import PreTrainedModel

from .gse import GSEConfig, GSEModel
from .utils.peft_types import TaskType

logger = logging.getLogger(__name__)


def get_gse_model(
    model: PreTrainedModel,
    gse_config: GSEConfig,
    adapter_name: str = "default",
    gse_cache_path: str = None,
) -> nn.Module:
    """
    Apply GSE (Generalized and Specialized Expert) to a model.

    When ``gse_cache_path`` is supplied the function implements a
    decomposition-cache workflow:

      1. If the cache file already exists, the model is built with
         ``skip_svd_init=True`` (cheap kaiming init, correct scaling)
         and the cached state_dict is loaded on top -- no SVD needed.
      2. Otherwise a normal SVD-based init is performed and the
         resulting full model state_dict is saved to ``gse_cache_path``
         so that the next run (training or inference) can skip SVD.

    Args:
        model: The base model to apply GSE to.
        gse_config: GSE configuration.
        adapter_name: Name of the adapter.
        gse_cache_path: Optional path to a ``.pt`` cache file for the
            post-SVD model state_dict.

    Returns:
        Model with GSE layers applied.
    """
    gse_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if gse_cache_path and os.path.isfile(gse_cache_path):
        logger.info(f"GSE cache found: {gse_cache_path}  -- skipping SVD decomposition")
        gse_config.skip_svd_init = True
        gse_model = GSEModel(model, {adapter_name: gse_config}, adapter_name)

        cached_state = torch.load(gse_cache_path, map_location="cpu")
        missing, unexpected = gse_model.load_state_dict(cached_state, strict=False)
        logger.info(f"GSE cache loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            for k in missing[:10]:
                logger.debug(f"  missing: {k}")
        if unexpected:
            for k in unexpected[:10]:
                logger.debug(f"  unexpected: {k}")
        del cached_state
        return gse_model

    gse_model = GSEModel(model, {adapter_name: gse_config}, adapter_name)

    if gse_cache_path:
        os.makedirs(os.path.dirname(gse_cache_path) or ".", exist_ok=True)
        logger.info(f"Saving GSE decomposition cache to: {gse_cache_path}")
        torch.save(gse_model.state_dict(), gse_cache_path)
        logger.info("GSE decomposition cache saved successfully")

    return gse_model


class GSEPeftModel(nn.Module):
    """
    GSE PEFT Model wrapper for easier integration.
    """
    
    def __init__(self, model: PreTrainedModel, gse_config: GSEConfig, adapter_name: str = "default"):
        super().__init__()
        self.base_model = GSEModel(model, {adapter_name: gse_config}, adapter_name)
        self.active_adapter = adapter_name
        self.peft_config = {adapter_name: gse_config}
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def get_base_model(self) -> nn.Module:
        """Return the base model"""
        return self.base_model.model
    
    def get_nb_trainable_parameters(self) -> tuple:
        """Return the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    
    def print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters"""
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable: {trainable_params / all_param:.2%}")
    
    def get_aux_loss(self, adapter_name: str = "default") -> torch.Tensor:
        """Get auxiliary loss from MoE layers (only for specialized experts)"""
        return self.base_model.get_aux_loss(adapter_name)
    
    def get_expert_info(self) -> dict:
        """Get information about generalized and specialized experts"""
        return self.base_model.get_expert_info()
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.base_model(*args, **kwargs)
