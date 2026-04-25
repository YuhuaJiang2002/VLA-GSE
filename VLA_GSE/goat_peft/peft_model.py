"""
GOAT PEFT Model wrapper
Simplified version for VLA-GSE integration
"""
import torch
from torch import nn
from transformers import PreTrainedModel

from .goat import GOATConfig, GOATModel
from .utils.peft_types import TaskType


def get_goat_model(
    model: PreTrainedModel, 
    goat_config: GOATConfig, 
    adapter_name: str = "default"
) -> nn.Module:
    """
    Apply GOAT (Gated MoE LoRA) to a model.
    
    Args:
        model: The base model to apply GOAT to
        goat_config: GOAT configuration
        adapter_name: Name of the adapter
    
    Returns:
        Model with GOAT layers applied
    """
    goat_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    
    # Create GOAT model wrapper
    goat_model = GOATModel(model, {adapter_name: goat_config}, adapter_name)
    
    return goat_model


class GOATPeftModel(nn.Module):
    """
    GOAT PEFT Model wrapper for easier integration.
    """
    
    def __init__(self, model: PreTrainedModel, goat_config: GOATConfig, adapter_name: str = "default"):
        super().__init__()
        self.base_model = GOATModel(model, {adapter_name: goat_config}, adapter_name)
        self.active_adapter = adapter_name
        self.peft_config = {adapter_name: goat_config}
    
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
        """Get auxiliary loss from MoE layers"""
        return self.base_model.get_aux_loss(adapter_name)
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.base_model(*args, **kwargs)
