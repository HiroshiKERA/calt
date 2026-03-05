from .base import ModelRegistry, get_model, get_model_from_config
from .bert.model import BertForSingleTokenClassification
from .generic.model import Transformer, TransformerConfig
from .gpt2.model import GPT2ForPromptedGeneration
from .loader import ModelLoader
from .pipeline import ModelPipeline


# Import loaders lazily to avoid circular imports
def _get_loaders():
    from .bart.loader import BartLoader
    from .bert.loader import BertLoader
    from .generic.loader import TransformerLoader
    from .gpt2.loader import GPT2Loader

    return TransformerLoader, BartLoader, GPT2Loader, BertLoader


__all__ = [
    "ModelRegistry",
    "get_model",
    "get_model_from_config",
    "ModelLoader",
    "ModelPipeline",
    "Transformer",
    "TransformerConfig",
    "GPT2ForPromptedGeneration",
    "BertForSingleTokenClassification",
]
