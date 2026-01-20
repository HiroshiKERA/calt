"""
Base class for model loaders.

Model loaders handle the conversion from unified config format (cfg.model) to
model-specific configs and create model instances.
"""

from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast


class ModelLoader(ABC):
    """Abstract base class for model loaders.
    
    Each model loader is responsible for:
    1. Converting unified config format (cfg.model) to model-specific config
    2. Creating the model instance from the converted config
    
    Attributes:
        calt_config (DictConfig): Original config from cfg.model (before conversion).
        config (PretrainedConfig): Model-specific config (after conversion).
        model (PreTrainedModel): Model instance.
    """
    
    def __init__(
        self,
        calt_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        """Initialize the model loader.
        
        Args:
            calt_config (DictConfig): Model configuration from cfg.model (OmegaConf).
            tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance (required for some models).
        """
        self.calt_config = calt_config
        self.tokenizer = tokenizer
        self.config: Optional[PretrainedConfig] = None
        self.model: Optional[PreTrainedModel] = None
    
    @abstractmethod
    def translate_config(self) -> PretrainedConfig:
        """Convert unified config format to model-specific config.
        
        This method should also set self.config to the translated config.
        
        Returns:
            PretrainedConfig: Model-specific configuration.
        """
        raise NotImplementedError
    
    @abstractmethod
    def build_model(self) -> PreTrainedModel:
        """Create model instance from the translated config.
        
        Returns:
            PreTrainedModel: Model instance.
        """
        raise NotImplementedError
    
    def load(self) -> PreTrainedModel:
        """Load the model by translating config and building the model.
        
        Returns:
            PreTrainedModel: Model instance.
        """
        self.config = self.translate_config()
        self.model = self.build_model()
        return self.model


def get_model_loader(
    model_name: Optional[str] = None,
    calt_config: Optional[DictConfig] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
) -> ModelLoader:
    """Create a model loader instance based on the specified name or config.
    
    Args:
        model_name (str | None): Name of the model type. If None, will be inferred from calt_config.
        calt_config (DictConfig | None): Model configuration from cfg.model (OmegaConf).
        tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance (required for some models).
    
    Returns:
        ModelLoader: Model loader instance.
    
    Raises:
        ValueError: If model_name is not supported or cannot be inferred from config.
    
    Examples:
        >>> # Create loader with explicit name and config
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("config/train.yaml")
        >>> loader = get_model_loader("transformer", cfg.model, tokenizer)
        >>> model = loader.load()
        >>> 
        >>> # Create loader from config only (model_type inferred from config)
        >>> loader = get_model_loader(calt_config=cfg.model, tokenizer=tokenizer)
        >>> model = loader.load()
    """
    # Import here to avoid circular import
    from .transformer import TransformerLoader
    from .bart import BartLoader
    
    # Determine model name
    if model_name is None:
        if calt_config is None:
            raise ValueError("Either model_name or calt_config must be provided")
        
        # Try to infer model name from config
        if hasattr(calt_config, 'model_type'):
            model_name = calt_config.model_type
        else:
            raise ValueError(
                "Cannot infer model_name from calt_config. "
                "Please provide model_name explicitly or set calt_config.model_type."
            )
    
    model_name = model_name.lower() if isinstance(model_name, str) else model_name
    
    # Create appropriate loader
    if model_name in ["transformer", "calt"]:
        return TransformerLoader(calt_config=calt_config, tokenizer=tokenizer)
    elif model_name == "bart":
        return BartLoader(calt_config=calt_config, tokenizer=tokenizer)
    else:
        supported_types = ["transformer", "calt", "bart"]
        raise ValueError(
            f"Unsupported model type: {model_name}. "
            f"Supported types: {supported_types}"
        )
