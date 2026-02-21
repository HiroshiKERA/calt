"""
Trainer pipeline for creating trainers from config.

This module provides a class-based interface similar to IOPipeline for creating
trainer instances from configuration files.
"""

import os
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from ..io.base import StandardDataCollator
from .trainer import Trainer


class TrainerPipeline:
    """Pipeline for creating trainers from configuration.

    Similar to IOPipeline, this class provides a simple interface for creating
    trainer instances from config files. It automatically selects the appropriate
    TrainerLoader based on the config.

    Examples:
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
        io_dict: Optional[dict] = None,
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
        # Prefer explicit arguments, but allow filling from io_dict when provided
        if io_dict is not None:
            tokenizer = tokenizer or io_dict.get("tokenizer")
            train_dataset = train_dataset or io_dict.get("train_dataset")
            eval_dataset = eval_dataset or io_dict.get("test_dataset")
            data_collator = data_collator or io_dict.get("data_collator")

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
            # "none" works in both HF v4 and v5 for no reporting
            if not hasattr(self.config, "report_to"):
                setattr(self.config, "report_to", "none")
            return

        os.environ.pop("WANDB_DISABLED", None)
        for key, env in [
            ("project", "WANDB_PROJECT"),
            ("group", "WANDB_RUN_GROUP"),
            ("name", "WANDB_NAME"),
        ]:
            value = getattr(wb, key, None)
            if value is not None:
                os.environ[env] = str(value)

        # Attach wandb config to training config so TrainerLoader can inspect it
        setattr(self.config, "wandb", wb)

        # Make sure Trainer sees wandb settings
        setattr(self.config, "report_to", getattr(self.config, "report_to", "wandb"))

    def build(self) -> "TrainerPipeline":
        """Build the trainer from configuration.

        Returns:
            TrainerPipeline: Returns self for method chaining.
        """
        # Configure wandb before building TrainingArguments / Trainer
        self._configure_wandb()

        # Import here to avoid circular import
        from .loader import StandardTrainerLoader

        # Create trainer loader
        self._loader = StandardTrainerLoader(
            calt_config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )

        # Load the trainer
        self.trainer = self._loader.load()
        return self

    def train(self, resume_from_checkpoint: str | bool | None = None) -> None:
        """Train the model.

        This method calls trainer.train(). If resume_from_checkpoint is provided,
        training will resume from the specified checkpoint.

        Args:
            resume_from_checkpoint: If True, resume from the latest checkpoint in output_dir.
                                   If str, resume from the specified checkpoint path.
                                   If None, start training from scratch.
        """
        if self.trainer is None:
            raise ValueError("Trainer not built. Call build() first.")

        # Train the model
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the model and tokenizer.

        Args:
            output_dir: Directory to save the model and tokenizer. If None, uses trainer's output_dir.
        """
        if self.trainer is None:
            raise ValueError("Trainer not built. Call build() first.")
        if output_dir is None:
            output_dir = self.trainer.args.output_dir
        self.trainer.save_model(output_dir=output_dir)
        # Also save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def evaluate_and_save_generation(self, max_length: int = 512) -> float:
        """Evaluate and save generation results."""
        if self.trainer is None:
            raise ValueError("Trainer not built. Call build() first.")
        return self.trainer.evaluate_and_save_generation(max_length=max_length)

    def __getattr__(self, name: str):
        """Delegate attribute access to trainer if not found in TrainerPipeline.

        This allows calling trainer methods directly on TrainerPipeline, e.g.:
        - trainer_pipeline.evaluate_and_save_generation()
        - trainer_pipeline.evaluate()
        - etc.
        """
        if self.trainer is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                "Trainer not built. Call build() first."
            )
        return getattr(self.trainer, name)

    @classmethod
    def from_io_dict(
        cls,
        config: DictConfig,
        model: PreTrainedModel,
        io_dict: dict,
        wandb_config: Optional[DictConfig] = None,
    ) -> "TrainerPipeline":
        """Create a TrainerPipeline from a dict returned by IOPipeline.build().

        Args:
            config: Training configuration (cfg.train). May contain wandb config as config.wandb.
            model: Model instance from ModelPipeline.
            io_dict: IOPipeline.build() result.
            wandb_config: Optional wandb configuration block. If None, tries to get from config.wandb.
        """
        # Get wandb_config from config.wandb if not provided
        if wandb_config is None and hasattr(config, "wandb"):
            wandb_config = config.wandb

        instance = cls(
            config=config,
            model=model,
            tokenizer=io_dict["tokenizer"],
            train_dataset=io_dict["train_dataset"],
            eval_dataset=io_dict["test_dataset"],
            data_collator=io_dict["data_collator"],
            wandb_config=wandb_config,
            io_dict=io_dict,
        )
        return instance

    @classmethod
    def resume_from_checkpoint(
        cls,
        save_dir: str,
        resume_from_checkpoint: bool = True,
    ) -> "TrainerPipeline":
        """Resume training from a saved checkpoint directory.

        This method loads train.yaml from save_dir, reconstructs IOPipeline, ModelPipeline,
        and TrainerPipeline, and optionally loads the saved model and tokenizer.

        Args:
            save_dir: Directory containing train.yaml, model/, and tokenizer/.
            resume_from_checkpoint: If True, load saved model and tokenizer. If False, create new model.

        Returns:
            TrainerPipeline: TrainerPipeline instance ready for training continuation.

        Examples:
            >>> from calt.trainer import TrainerPipeline
            >>>
            >>> # Load from checkpoint and continue training
            >>> trainer_pipeline = TrainerPipeline.resume_from_checkpoint("./results")
            >>> trainer_pipeline.build()
            >>> trainer_pipeline.train()  # Continue training
        """
        from .utils import load_from_checkpoint

        _, _, trainer_pipeline = load_from_checkpoint(save_dir, resume_from_checkpoint)
        return trainer_pipeline
