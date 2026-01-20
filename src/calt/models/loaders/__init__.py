"""
Model loaders for creating model instances from unified config format.

Each loader handles the conversion from cfg.model (OmegaConf) to model-specific configs
and creates the corresponding model instance.
"""

from .base import ModelLoader
from .transformer import TransformerLoader
from .bart import BartLoader

__all__ = [
    "ModelLoader",
    "TransformerLoader",
    "BartLoader",
]
