"""
GSE (Generalized and Specialized Expert) Configuration
"""
from dataclasses import dataclass, field
from peft import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class GSEConfig(LoraConfig):
    """
    GSE Configuration extending LoraConfig with Generalized and Specialized Expert parameters.
    
    Args:
        num_experts: Total number of experts (generalized + specialized) (default: 8)
        num_generalized_experts: Number of generalized experts that are always selected (default: 2)
        top_k: The k in top-k gating for specialized experts (default: 2)
        init_type: Initialization type ("gse", "goat", "lora", etc.)
        init_cof: Initialization coefficient (default: 1.0)
        specialized_scaling_method: Specialized expert scaling method.
            "default" preserves the existing implementation. Use
            "gradient_scale_balancing" to set specialized expert scales with
            the trace-inverse rule from the Gradient Scale Balancing analysis.
    
    Note:
        - Generalized experts are always activated and initialized from the largest singular values
        - Specialized experts are selected by the router and initialized from remaining singular values
        - num_specialized_experts = num_experts - num_generalized_experts
    """
    num_experts: int = field(default=8, metadata={"help": "Total number of experts (generalized + specialized)."})
    num_generalized_experts: int = field(default=2, metadata={"help": "Number of generalized experts (always selected)."})
    top_k: int = field(default=2, metadata={"help": "The k in top-k gating for specialized experts"})
    init_type: str = field(default="gse", metadata={"help": "Initialization type"})
    init_cof: float = field(default=1.0, metadata={"help": "Initialization coefficient"})
    specialized_scaling_method: str = field(
        default="default",
        metadata={"help": "Specialized expert scaling method: 'default' or "
                          "'gradient_scale_balancing'."})
    specialized_scaling_base: float = field(
        default=2.0,
        metadata={"help": "Base scale s_base for gradient_scale_balancing specialized scaling."})
    specialized_scaling_eps: float = field(
        default=1e-12,
        metadata={"help": "Numerical epsilon for trace-inverse specialized scaling."})
    skip_svd_init: bool = field(
        default=False,
        metadata={"help": "Skip SVD computation during init (structure + scaling preserved). "
                          "Use when loading from a cached decomposition or checkpoint."})
    aux_loss_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for load-balancing auxiliary loss. "
                          "Set to 0 to disable aux loss computation entirely in the forward pass."})

    def __post_init__(self):
        self.peft_type = PeftType.GSE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # Validate configuration
        if self.num_generalized_experts >= self.num_experts:
            raise ValueError(
                f"num_generalized_experts ({self.num_generalized_experts}) must be less than "
                f"num_experts ({self.num_experts})"
            )
        if self.num_generalized_experts < 0:
            raise ValueError(f"num_generalized_experts must be non-negative, got {self.num_generalized_experts}")
        self.specialized_scaling_method = str(self.specialized_scaling_method).lower()
        valid_scaling_methods = {"default", "gradient_scale_balancing", "gsb", "trace_inverse"}
        if self.specialized_scaling_method not in valid_scaling_methods:
            raise ValueError(
                f"specialized_scaling_method must be one of {sorted(valid_scaling_methods)}, "
                f"got {self.specialized_scaling_method!r}"
            )
