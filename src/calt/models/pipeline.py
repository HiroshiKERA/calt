"""
Model pipeline for creating models from config.

This module provides a class-based interface similar to IOPipeline for creating
model instances from configuration files.
"""

from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .loaders.base import get_model_loader


class ModelPipeline:
    """Pipeline for creating models from configuration.
    
    Similar to IOPipeline, this class provides a simple interface for creating
    model instances from config files. It automatically selects the appropriate
    ModelLoader based on the config.
    
    Example:
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
        self._loader = None
    
    def build(self) -> PreTrainedModel:
        """Build the model from configuration.
        
        Returns:
            PreTrainedModel: Model instance.
        """
        # Get the appropriate loader
        self._loader = get_model_loader(
            calt_config=self.calt_config,
            tokenizer=self.tokenizer,
        )
        
        # Load the model
        self.model = self._loader.load()
        return self.model
