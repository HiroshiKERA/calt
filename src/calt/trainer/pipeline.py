"""
Trainer pipeline for creating trainers from config.

This module provides a class-based interface similar to IOPipeline for creating
trainer instances from configuration files.
"""

from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from torch.utils.data import Dataset

from ..io.base import StandardDataCollator
from .loaders.base import get_trainer_loader
from .trainer import Trainer


class TrainerPipeline:
    """Pipeline for creating trainers from configuration.
    
    Similar to IOPipeline, this class provides a simple interface for creating
    trainer instances from config files. It automatically selects the appropriate
    TrainerLoader based on the config.
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> from calt.trainer import TrainerPipeline
        >>> 
        >>> cfg = OmegaConf.load("config/train.yaml")
        >>> model = ...  # Get model from ModelPipeline
        >>> tokenizer = ...  # Get tokenizer from IOPipeline
        >>> train_dataset = ...  # Get from IOPipeline
        >>> eval_dataset = ...  # Get from IOPipeline
        >>> data_collator = ...  # Get from IOPipeline
        >>> 
        >>> trainer_pipeline = TrainerPipeline(
        ...     cfg.train,
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     data_collator=data_collator,
        ... )
        >>> trainer = trainer_pipeline.build()
    """
    
    def __init__(
        self,
        calt_config: DictConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[StandardDataCollator] = None,
    ):
        """Initialize the trainer pipeline.
        
        Args:
            calt_config (DictConfig): Training configuration from cfg.train (OmegaConf).
            model (PreTrainedModel | None): Model instance.
            tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance.
            train_dataset (Dataset | None): Training dataset.
            eval_dataset (Dataset | None): Evaluation dataset.
            data_collator (StandardDataCollator | None): Data collator.
        """
        self.calt_config = calt_config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.trainer: Optional[Trainer] = None
        self._loader = None
    
    def build(self) -> Trainer:
        """Build the trainer from configuration.
        
        Returns:
            Trainer: Trainer instance.
        """
        # Get the appropriate loader
        self._loader = get_trainer_loader(
            calt_config=self.calt_config,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )
        
        # Load the trainer
        self.trainer = self._loader.load()
        return self.trainer
