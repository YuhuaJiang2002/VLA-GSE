"""
GSE Model - Generalized and Specialized Expert
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from peft import LoraModel

from .config import GSEConfig
from .layer import GSELayer, LinearGSELayer


class GSEModel(LoraModel):
    """
    GSE (Generalized and Specialized Expert) Model.
    Extends LoraModel with MoE (Mixture of Experts) gating.
    
    Key features:
    - Generalized experts: Always activated, initialized from largest singular values
    - Specialized experts: Selected by top-k router, initialized from remaining singular values
    """
    prefix: str = "lora_"
        
    def _create_and_replace(
        self, gse_config: GSEConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """Create and replace target module with GSE layer"""
        kwargs = {
            "lora_rank": gse_config.r,
            "lora_alpha": gse_config.lora_alpha,
            "lora_dropout": gse_config.lora_dropout,
            "init_lora_weights": gse_config.init_lora_weights,
            "num_experts": gse_config.num_experts,
            "num_generalized_experts": gse_config.num_generalized_experts,
            "top_k": gse_config.top_k,
            "init_type": gse_config.init_type,
            "init_cof": gse_config.init_cof,
            "specialized_scaling_method": gse_config.specialized_scaling_method,
            "specialized_scaling_base": gse_config.specialized_scaling_base,
            "specialized_scaling_eps": gse_config.specialized_scaling_eps,
            "skip_svd_init": gse_config.skip_svd_init,
            "aux_loss_weight": gse_config.aux_loss_weight,
        }

        if isinstance(target, GSELayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        """Create new GSE module to replace target"""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            # Skip if rank is larger than the minimum dimension
            if min(target.weight.shape[0], target.weight.shape[1]) < kwargs['lora_rank']:
                return target
            new_module = LinearGSELayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module

    def get_aux_loss(self, adapter_name="default") -> torch.Tensor:
        """Get auxiliary loss from all MoE layers (load balancing loss for specialized experts)"""
        model_loss = torch.tensor(0, dtype=torch.float).to(self.model.device)
        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                layer_loss = module[adapter_name].layer_loss
                if layer_loss is not None:
                    model_loss += layer_loss
        return model_loss

    def _set_adapter_layers(self, enabled=True):
        """Enable or disable adapter layers"""
        for module in self.model.modules():
            if isinstance(module, GSELayer):
                module.disable_adapters = False if enabled else True

    def set_adapter(self, adapter_name="default", inference_mode=False):
        """Set active adapter
        
        Args:
            adapter_name: Name of the adapter to set as active
            inference_mode: Whether to set the adapter in inference mode (ignored, kept for compatibility)
        """
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper
        
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]
        
        _adapters_has_been_set = False
        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeed in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def enable_adapter_layers(self):
        """Enable all adapter layers"""
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        """Disable all adapter layers"""
        self._set_adapter_layers(enabled=False)

    def get_nb_trainable_parameters(self) -> tuple:
        """Return the number of trainable parameters and total parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters in the model"""
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable: {trainable_params / all_param:.2%}"
        )
    
    def get_expert_info(self) -> dict:
        """Get information about generalized and specialized experts"""
        info = {
            "total_generalized_experts": 0,
            "total_specialized_experts": 0,
            "layers": []
        }
        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                for adapter_name, moe in module.items():
                    num_gen = len(moe.generalized_experts)
                    num_spec = len(moe.specialized_experts)
                    info["total_generalized_experts"] += num_gen
                    info["total_specialized_experts"] += num_spec
                    info["layers"].append({
                        "name": name,
                        "adapter": adapter_name,
                        "generalized_experts": num_gen,
                        "specialized_experts": num_spec
                    })
        return info
