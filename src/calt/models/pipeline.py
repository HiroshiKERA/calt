"""
Model pipeline for creating models from config.

This module provides a class-based interface similar to IOPipeline for creating
model instances from configuration files using ModelRegistry.
"""

from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .base import ModelRegistry


class ModelPipeline:
    """Pipeline for creating models from configuration using ModelRegistry.

    Similar to IOPipeline, this class provides a simple interface for creating
    model instances from config files. It uses ModelRegistry internally to handle
    model creation.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> from calt.models import ModelPipeline
        >>>
        >>> cfg = OmegaConf.load("config/train.yaml")
        >>> tokenizer = ...  # Get tokenizer from IOPipeline
        >>>
        >>> model_pipeline = ModelPipeline(cfg.model, tokenizer)
        >>> model = model_pipeline.build()
    """

    def __init__(
        self,
        calt_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        """Initialize the model pipeline.

        Args:
            calt_config (DictConfig): Model configuration from cfg.model (OmegaConf).
            tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance (required for some models).
        """
        self.calt_config = calt_config
        self.tokenizer = tokenizer
        self.model: Optional[PreTrainedModel] = None
        self._registry = ModelRegistry()

    @classmethod
    def from_io_dict(
        cls,
        calt_config: DictConfig,
        io_dict: dict,
    ) -> "ModelPipeline":
        """Create a ModelPipeline using the result dict from IOPipeline.build().

        Args:
            calt_config: Model configuration (cfg.model).
            io_dict: Result dict from ``IOPipeline.build()``, expected to contain
                at least the ``\"tokenizer\"`` entry.
        """
        return cls(
            calt_config=calt_config,
            tokenizer=io_dict["tokenizer"],
        )

    def build(self) -> PreTrainedModel:
        """Build the model from configuration using ModelRegistry.

        Returns:
            PreTrainedModel: Model instance.
        """
        # Use ModelRegistry to create the model
        self.model = self._registry.create_from_config(
            model_config=self.calt_config,
            tokenizer=self.tokenizer,
        )
        return self.model
