from .base import ModelRegistry, get_model, get_model_from_config
from .loaders import ModelLoader, get_model_loader
from .pipeline import ModelPipeline
from .generic.model import Transformer, TransformerConfig

# Import loaders lazily to avoid circular imports
def _get_loaders():
    from .generic.loader import TransformerLoader
    from .bart.loader import BartLoader
    return TransformerLoader, BartLoader

__all__ = [
    "ModelRegistry",
    "get_model",
    "get_model_from_config",
    "ModelLoader",
    "get_model_loader",
    "ModelPipeline",
    "Transformer",
    "TransformerConfig",
]
