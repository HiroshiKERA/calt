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
