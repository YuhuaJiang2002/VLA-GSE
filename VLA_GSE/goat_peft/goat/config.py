"""
GOAT Configuration
"""
from dataclasses import dataclass, field
from peft import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class GOATConfig(LoraConfig):
    """
    GOAT Configuration extending LoraConfig with MoE parameters.
    
    Args:
        num_experts: Number of experts in MoE (default: 8)
        top_k: The k in top-k gating (default: 2)
        init_type: Initialization type for GOAT ("goat", "lora", etc.)
        init_cof: Initialization coefficient (default: 1.0)
    """
    num_experts: int = field(default=8, metadata={"help": "The number of experts in MoE."})
    top_k: int = field(default=2, metadata={"help": "The k in top-k gating"})
    init_type: str = field(default="goat", metadata={"help": "Initialization type"})
    init_cof: float = field(default=1.0, metadata={"help": "Initialization coefficient"})

    def __post_init__(self):
        self.peft_type = PeftType.GOAT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
