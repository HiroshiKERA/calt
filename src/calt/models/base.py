"""
Model registry for creating model instances.

This module provides a registry class for creating different types of models.
"""

from typing import Callable, Optional, Type

from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast


class ModelRegistry:
    """Registry for creating model instances based on model type.

    This class provides a unified interface for creating different types of models.
    Models can be registered and retrieved by name or inferred from config.

    Examples:
        >>> # Create model with explicit name and config
        >>> from calt.models.generic import TransformerConfig
        >>> registry = ModelRegistry()
        >>> config = TransformerConfig(vocab_size=1000, d_model=128)
        >>> model = registry.create("transformer", config)
        >>>
        >>> # Create model from config only (model_type inferred from config)
        >>> model = registry.create(model_config=config)
        >>>
        >>> # Register custom model
        >>> registry.register("custom_model", CustomModel, CustomModelConfig)
    """

    def __init__(self):
        """Initialize the registry with default model types."""
        self._registry: dict[
            str, tuple[Type[PreTrainedModel], Type[PretrainedConfig]]
        ] = {}
        self._config_mappings: dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default model types."""
        # Import here to avoid circular import
        from transformers import BartConfig, BartForConditionalGeneration

        from .bart.config_mapping import create_bart_config
        from .generic.config_mapping import create_transformer_config
        from .generic.model import Transformer, TransformerConfig

        # Register generic transformer model (aliases: transformer, calt, generic)
        self.register("transformer", Transformer, TransformerConfig)
        self.register("calt", Transformer, TransformerConfig)
        self.register("generic", Transformer, TransformerConfig)
        self.register_config_mapping("transformer", create_transformer_config)
        self.register_config_mapping("calt", create_transformer_config)
        self.register_config_mapping("generic", create_transformer_config)

        # Register BART model
        self.register("bart", BartForConditionalGeneration, BartConfig)
        self.register_config_mapping("bart", create_bart_config)

    def register(
        self,
        model_name: str,
        model_class: Type[PreTrainedModel],
        config_class: Type[PretrainedConfig],
    ):
        """Register a model class with the registry.

        Args:
            model_name (str): Name to register the model under.
            model_class (Type[PreTrainedModel]): Model class to register.
            config_class (Type[PretrainedConfig]): Config class for the model.
        """
        self._registry[model_name.lower()] = (model_class, config_class)

    def register_config_mapping(
        self,
        model_name: str,
        mapping_func: Callable[
            [DictConfig, Optional[PreTrainedTokenizerFast]], PretrainedConfig
        ],
    ):
        """Register a config mapping function for converting OmegaConf to model config.

        Args:
            model_name (str): Name of the model type.
            mapping_func (Callable): Function that takes (model_config: DictConfig, tokenizer: Optional)
                and returns PretrainedConfig.
        """
        self._config_mappings[model_name.lower()] = mapping_func

    def create(
        self,
        model_name: Optional[str] = None,
        model_config: Optional[PretrainedConfig] = None,
    ) -> PreTrainedModel:
        """Create a model instance based on the specified name or config.

        Args:
            model_name (str | None): Name of the model type. If None, will be inferred from model_config.
            model_config (PretrainedConfig | None): Model configuration. If provided and contains
                model_type attribute, model_name can be inferred from it.

        Returns:
            PreTrainedModel: Model instance.

        Raises:
            ValueError: If model_name is not supported or cannot be inferred from config.

        Examples:
            >>> # Create model with explicit name and config
            >>> from calt.models.generic import TransformerConfig
            >>> registry = ModelRegistry()
            >>> config = TransformerConfig(vocab_size=1000, d_model=128)
            >>> model = registry.create("transformer", config)
            >>>
            >>> # Create model from config only (model_type inferred from config)
            >>> model = registry.create(model_config=config)
        """
        # Determine model name
        if model_name is None:
            if model_config is None:
                raise ValueError("Either model_name or model_config must be provided")

            # Try to infer model name from config
            if hasattr(model_config, "model_type"):
                model_name = model_config.model_type
            else:
                raise ValueError(
                    "Cannot infer model_name from model_config. "
                    "Please provide model_name explicitly or ensure model_config has model_type attribute."
                )

        model_name = model_name.lower() if isinstance(model_name, str) else model_name

        # Get model class from registry
        if model_name not in self._registry:
            supported_types = list(self._registry.keys())
            raise ValueError(
                f"Unsupported model type: {model_name}. "
                f"Supported types: {supported_types}"
            )

        model_class, config_class = self._registry[model_name]

        # Validate config type
        if model_config is None:
            raise ValueError(f"model_config is required for {model_name} model")

        if not isinstance(model_config, config_class):
            raise ValueError(
                f"model_config must be {config_class.__name__} for {model_name} model, "
                f"got {type(model_config).__name__}"
            )

        return model_class(config=model_config)

    def create_from_config(
        self,
        model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        model_name: Optional[str] = None,
    ) -> PreTrainedModel:
        """Create a model instance from OmegaConf config (cfg.model).

        This method automatically converts the unified config format to model-specific configs
        using registered config mapping functions.

        Args:
            model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
            tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance (required for some models like BART).
            model_name (str | None): Name of the model type. If None, will be inferred from model_config.model_type.

        Returns:
            PreTrainedModel: Model instance.

        Raises:
            ValueError: If model_name is not supported or cannot be inferred from config.

        Examples:
            >>> # Create model from OmegaConf config
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.load("config/train.yaml")
            >>> registry = ModelRegistry()
            >>> model = registry.create_from_config(cfg.model, tokenizer)
        """
        # Determine model name
        if model_name is None:
            if hasattr(model_config, "model_type"):
                model_name = model_config.model_type
            else:
                raise ValueError(
                    "Cannot infer model_name from model_config. "
                    "Please provide model_name explicitly or set model_config.model_type."
                )

        model_name = model_name.lower() if isinstance(model_name, str) else model_name

        # Get model class from registry
        if model_name not in self._registry:
            supported_types = list(self._registry.keys())
            raise ValueError(
                f"Unsupported model type: {model_name}. "
                f"Supported types: {supported_types}"
            )

        model_class, config_class = self._registry[model_name]

        # Get config mapping function
        if model_name not in self._config_mappings:
            raise ValueError(
                f"No config mapping registered for model type: {model_name}. "
                f"Please register a config mapping using register_config_mapping()."
            )

        mapping_func = self._config_mappings[model_name]

        # Convert OmegaConf config to model-specific config
        converted_config = mapping_func(model_config, tokenizer)

        # Create and return model
        return model_class(config=converted_config)

    def list_models(self) -> list[str]:
        """List all registered model types.

        Returns:
            list[str]: List of registered model type names.
        """
        return list(self._registry.keys())


# Global registry instance
_registry = ModelRegistry()


def get_model(
    model_name: Optional[str] = None,
    model_config: Optional[PretrainedConfig] = None,
) -> PreTrainedModel:
    """Create a model instance using the global registry.

    This is a convenience function that uses the global ModelRegistry instance.

    Args:
        model_name (str | None): Name of the model type. If None, will be inferred from model_config.
        model_config (PretrainedConfig | None): Model configuration.

    Returns:
        PreTrainedModel: Model instance.

    Examples:
        >>> # Create model with explicit name and config
        >>> from calt.models.generic import TransformerConfig
        >>> config = TransformerConfig(vocab_size=1000, d_model=128)
        >>> model = get_model("transformer", config)
        >>>
        >>> # Create model from config only (model_type inferred from config)
        >>> model = get_model(model_config=config)
    """
    return _registry.create(model_name=model_name, model_config=model_config)


def get_model_from_config(
    model_config: DictConfig,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    model_name: Optional[str] = None,
) -> PreTrainedModel:
    """Create a model instance from OmegaConf config using ModelRegistry.

    This function uses the ModelRegistry to convert configs and create models.

    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance (required for some models like BART).
        model_name (str | None): Name of the model type. If None, will be inferred from model_config.model_type.

    Returns:
        PreTrainedModel: Model instance.

    Examples:
        >>> # Create model from OmegaConf config
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("config/train.yaml")
        >>> model = get_model_from_config(cfg.model, tokenizer)
    """
    return _registry.create_from_config(
        model_config=model_config,
        tokenizer=tokenizer,
        model_name=model_name,
    )
