"""
Trainer loader for creating trainer instances from config.

Handles conversion from unified config format (cfg.train) to TrainingArguments
and creates Trainer instances.
"""

from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast, TrainingArguments

from ..io.base import StandardDataCollator
from .trainer import Trainer
from .utils import count_cuda_devices

# Default TrainingArguments for reference
_DEFAULT_TRAINING_ARGS = TrainingArguments(output_dir="./tmp")


class TrainerLoader(ABC):
    """Abstract base class for trainer loaders.

    Each trainer loader is responsible for:
    1. Converting unified config format (cfg.train) to TrainingArguments
    2. Creating the trainer instance from the converted arguments

    Attributes:
        calt_config (DictConfig): Original config from cfg.train (before conversion).
        training_args (TrainingArguments): Training arguments (after conversion).
        trainer (Trainer): Trainer instance.
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
        """Initialize the trainer loader.

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
        self.training_args: Optional[TrainingArguments] = None
        self.trainer: Optional[Trainer] = None

    @abstractmethod
    def translate_config(self) -> TrainingArguments:
        """Convert unified config format to TrainingArguments.

        Returns:
            TrainingArguments: Training arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def build_trainer(self) -> Trainer:
        """Create trainer instance from the translated arguments.

        Returns:
            Trainer: Trainer instance.
        """
        raise NotImplementedError

    def load(self) -> Trainer:
        """Load the trainer by translating config and building the trainer.

        Returns:
            Trainer: Trainer instance.
        """
        self.training_args = self.translate_config()
        self.trainer = self.build_trainer()
        return self.trainer


class StandardTrainerLoader(TrainerLoader):
    """Loader for standard Trainer.

    Converts cfg.train to TrainingArguments and creates Trainer instances.
    """

    def translate_config(self) -> TrainingArguments:
        """Convert unified config format to TrainingArguments.

        Returns:
            TrainingArguments: Training arguments.
        """
        # Calculate batch sizes (device-aware)
        if self.model is not None and hasattr(self.model, "device"):
            # If model is on CUDA, divide batch size by number of devices
            import torch

            if torch.cuda.is_available():
                train_batch_size = self.calt_config.batch_size // count_cuda_devices()
                test_batch_size = (
                    self.calt_config.test_batch_size // count_cuda_devices()
                )
            else:
                train_batch_size = self.calt_config.batch_size
                test_batch_size = self.calt_config.test_batch_size
        else:
            # Fallback: assume single device
            train_batch_size = self.calt_config.batch_size
            test_batch_size = self.calt_config.test_batch_size

        # Determine lr_scheduler_type
        lr_scheduler_type = (
            "constant"
            if getattr(self.calt_config, "lr_scheduler_type", "linear") == "constant"
            else "linear"
        )

        # Determine report_to (wandb or None)
        report_to = getattr(self.calt_config, "report_to", "wandb")
        if hasattr(self.calt_config, "wandb") and getattr(
            self.calt_config.wandb, "no_wandb", False
        ):
            report_to = None

        self.training_args = TrainingArguments(
            output_dir=self.calt_config.get(
                "save_dir", self.calt_config.get("output_dir", "./tmp")
            ),
            num_train_epochs=self.calt_config.num_train_epochs,
            learning_rate=self.calt_config.learning_rate,
            weight_decay=getattr(
                self.calt_config, "weight_decay", _DEFAULT_TRAINING_ARGS.weight_decay
            ),
            warmup_ratio=getattr(
                self.calt_config, "warmup_ratio", _DEFAULT_TRAINING_ARGS.warmup_ratio
            ),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=test_batch_size,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=getattr(
                self.calt_config, "max_grad_norm", _DEFAULT_TRAINING_ARGS.max_grad_norm
            ),
            optim=getattr(self.calt_config, "optimizer", _DEFAULT_TRAINING_ARGS.optim),
            # Dataloader settings
            dataloader_num_workers=getattr(
                self.calt_config,
                "num_workers",
                _DEFAULT_TRAINING_ARGS.dataloader_num_workers,
            ),
            dataloader_pin_memory=getattr(
                self.calt_config, "dataloader_pin_memory", True
            ),
            # Evaluation and saving settings
            eval_strategy=getattr(self.calt_config, "eval_strategy", "steps"),
            eval_steps=getattr(self.calt_config, "eval_steps", 1000),
            save_strategy=getattr(self.calt_config, "save_strategy", "steps"),
            save_steps=getattr(self.calt_config, "save_steps", 1000),
            save_total_limit=getattr(self.calt_config, "save_total_limit", 1),
            label_names=getattr(self.calt_config, "label_names", ["labels"]),
            save_safetensors=getattr(self.calt_config, "save_safetensors", False),
            # Logging settings
            logging_strategy=getattr(self.calt_config, "logging_strategy", "steps"),
            logging_steps=getattr(self.calt_config, "logging_steps", 50),
            report_to=report_to,
            run_name=getattr(self.calt_config, "run_name", None),
            # Others
            remove_unused_columns=getattr(
                self.calt_config, "remove_unused_columns", False
            ),
            seed=getattr(self.calt_config, "seed", _DEFAULT_TRAINING_ARGS.seed),
            disable_tqdm=getattr(self.calt_config, "disable_tqdm", True),
        )
        return self.training_args

    def build_trainer(self) -> Trainer:
        """Create Trainer instance.

        Returns:
            Trainer: Trainer instance.

        Raises:
            ValueError: If required components are not provided.
        """
        if self.training_args is None:
            raise ValueError(
                "training_args must be set before building trainer. Call translate_config() first."
            )
        if self.model is None:
            raise ValueError("model is required to build trainer")
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to build trainer")
        if self.train_dataset is None:
            raise ValueError("train_dataset is required to build trainer")
        if self.data_collator is None:
            raise ValueError("data_collator is required to build trainer")

        return Trainer(
            args=self.training_args,
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
