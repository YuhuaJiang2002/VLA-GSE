"""
PEFT and Task Types for GOAT
"""
import enum


class PeftType(str, enum.Enum):
    """PEFT Adapter Types"""
    LORA = "LORA"
    GOAT = "GOAT"


class TaskType(str, enum.Enum):
    """PEFT Task Type"""
    CAUSAL_LM = "CAUSAL_LM"
