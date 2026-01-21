"""
Base class for trainer loaders.

Trainer loaders handle the conversion from unified config format (cfg.train) to
TrainingArguments and create trainer instances.
"""

from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast, TrainingArguments

from ...io.base import StandardDataCollator
from ..trainer import Trainer


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


def get_trainer_loader(
    calt_config: DictConfig,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    data_collator: Optional[StandardDataCollator] = None,
) -> TrainerLoader:
    """Create a trainer loader instance.
    
    Args:
        calt_config (DictConfig): Training configuration from cfg.train (OmegaConf).
        model (PreTrainedModel | None): Model instance.
        tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance.
        train_dataset (Dataset | None): Training dataset.
        eval_dataset (Dataset | None): Evaluation dataset.
        data_collator (StandardDataCollator | None): Data collator.
    
    Returns:
        TrainerLoader: Trainer loader instance.
    
    Examples:
        >>> # Create loader with config
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("config/train.yaml")
        >>> loader = get_trainer_loader(cfg.train, model, tokenizer, train_dataset, eval_dataset, data_collator)
        >>> trainer = loader.load()
    """
    # Import here to avoid circular import
    from .trainer import StandardTrainerLoader
    
    return StandardTrainerLoader(
        calt_config=calt_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
