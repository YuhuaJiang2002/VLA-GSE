"""
PEFT and Task Types for GSE (Generalized and Specialized Expert)
"""
import enum


class PeftType(str, enum.Enum):
    """PEFT Adapter Types"""
    LORA = "LORA"
    GOAT = "GOAT"
    GSE = "GSE"


class TaskType(str, enum.Enum):
    """PEFT Task Type"""
    CAUSAL_LM = "CAUSAL_LM"
