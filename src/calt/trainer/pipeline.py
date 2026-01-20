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

from ..io.base import StandardDataCollator, IOPipeline
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
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[StandardDataCollator] = None,
        wandb_config: Optional[DictConfig] = None,
        io_pipeline: Optional[dict] = None,
    ):
        """Initialize the trainer pipeline.
        
        Args:
            config (DictConfig): Training configuration from cfg.train (OmegaConf).
            model (PreTrainedModel | None): Model instance.
            tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance.
            train_dataset (Dataset | None): Training dataset.
            eval_dataset (Dataset | None): Evaluation dataset.
            data_collator (StandardDataCollator | None): Data collator.
        """
        self.config = config
        self.model = model
        # Prefer explicit arguments, but allow filling from io_pipeline when provided
        if io_pipeline is not None:
            tokenizer = tokenizer or io_pipeline.get("tokenizer")
            train_dataset = train_dataset or io_pipeline.get("train_dataset")
            eval_dataset = eval_dataset or io_pipeline.get("test_dataset")
            data_collator = data_collator or io_pipeline.get("data_collator")

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
            setattr(self.config, "report_to", None)
            return

        # Enable wandb
        os.environ.pop("WANDB_DISABLED", None)

        project = getattr(wb, "project", 'calt')
        group = getattr(wb, "group", None)
        name = getattr(wb, "name", None)

        if project is not None:
            os.environ["WANDB_PROJECT"] = str(project)
        if group is not None:
            os.environ["WANDB_RUN_GROUP"] = str(group)
        if name is not None:
            os.environ["WANDB_NAME"] = str(name)

        # Attach wandb config to training config so TrainerLoader can inspect it
        setattr(self.config, "wandb", wb)

        # Make sure Trainer sees wandb settings
        setattr(self.config, "report_to", getattr(self.config, "report_to", "wandb"))
        if name is not None and not hasattr(self.config, "run_name"):
            setattr(self.config, "run_name", str(name))

    def build(self) -> Trainer:
        """Build the trainer from configuration.
        
        Returns:
            Trainer: Trainer instance.
        """
        # Configure wandb before building TrainingArguments / Trainer
        self._configure_wandb()

        # Get the appropriate loader
        self._loader = get_trainer_loader(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )
        
        # Load the trainer
        self.trainer = self._loader.load()
        return self.trainer

    @classmethod
    def from_io_settings(
        cls,
        config: DictConfig,
        model: PreTrainedModel,
        io_settings: dict,
        wandb_config: Optional[DictConfig] = None,
    ) -> "TrainerPipeline":
        """Create a TrainerPipeline from io_settings.
        
        Args:
            config: Training configuration (cfg.train).
            model: Model instance from ModelPipeline.
            io_settings: IOPipeline.build() result.
            wandb_config: Optional wandb configuration block (cfg.wandb).
        """
        return cls(
            config=config,
            model=model,
            tokenizer=io_settings["tokenizer"],
            train_dataset=io_settings["train_dataset"],
            eval_dataset=io_settings["test_dataset"],
            data_collator=io_settings["data_collator"],
            wandb_config=wandb_config,
        )