"""
Trainer loaders for creating trainer instances from unified config format.

Each loader handles the conversion from cfg.train (OmegaConf) to TrainingArguments
and creates the corresponding trainer instance.
"""

from .base import TrainerLoader, get_trainer_loader
from .trainer import StandardTrainerLoader

__all__ = [
    "TrainerLoader",
    "get_trainer_loader",
    "StandardTrainerLoader",
]
