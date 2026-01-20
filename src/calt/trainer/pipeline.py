"""
Trainer pipeline for creating trainers from config.

This module provides a class-based interface similar to IOPipeline for creating
trainer instances from configuration files.
"""

from typing import Optional
import os

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
        wandb_config: Optional[DictConfig] = None,
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
        self.wandb_config = wandb_config
        self.trainer: Optional[Trainer] = None
        self._loader = None

    def _configure_wandb(self) -> None:
        """Configure wandb-related settings (env vars and training config)."""
        if self.wandb_config is None:
            return

        wb = self.wandb_config
        no_wandb = getattr(wb, "no_wandb", False)

        if no_wandb:
            os.environ["WANDB_DISABLED"] = "true"
            # Explicitly disable reporting
            setattr(self.calt_config, "report_to", None)
            return

        # Enable wandb
        os.environ.pop("WANDB_DISABLED", None)

        project = getattr(wb, "project", None)
        group = getattr(wb, "group", None)
        name = getattr(wb, "name", None)

        if project is not None:
            os.environ["WANDB_PROJECT"] = str(project)
        if group is not None:
            os.environ["WANDB_RUN_GROUP"] = str(group)
        if name is not None:
            os.environ["WANDB_NAME"] = str(name)

        # Attach wandb config to training config so TrainerLoader can inspect it
        setattr(self.calt_config, "wandb", wb)

        # Make sure Trainer sees wandb settings
        setattr(self.calt_config, "report_to", getattr(self.calt_config, "report_to", "wandb"))
        if name is not None and not hasattr(self.calt_config, "run_name"):
            setattr(self.calt_config, "run_name", str(name))

    def build(self) -> Trainer:
        """Build the trainer from configuration.
        
        Returns:
            Trainer: Trainer instance.
        """
        # Configure wandb before building TrainingArguments / Trainer
        self._configure_wandb()

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
