"""
Loader for standard Trainer.

Handles conversion from unified config format (cfg.train) to TrainingArguments
and creates Trainer instances.
"""

from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast, TrainingArguments
from torch.utils.data import Dataset

from ...io.base import StandardDataCollator
from ...trainer.utils import count_cuda_devices
from ..trainer import Trainer
from .base import TrainerLoader

# Default TrainingArguments for reference
_DEFAULT_TRAINING_ARGS = TrainingArguments(output_dir="./tmp")


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
        if self.model is not None and hasattr(self.model, 'device'):
            # If model is on CUDA, divide batch size by number of devices
            import torch
            if torch.cuda.is_available():
                train_batch_size = self.calt_config.batch_size // count_cuda_devices()
                test_batch_size = self.calt_config.test_batch_size // count_cuda_devices()
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
            if getattr(self.calt_config, 'lr_scheduler_type', 'linear') == "constant"
            else "linear"
        )
        
        # Determine report_to (wandb or None)
        report_to = getattr(self.calt_config, 'report_to', 'wandb')
        if hasattr(self.calt_config, 'wandb') and getattr(self.calt_config.wandb, 'no_wandb', False):
            report_to = None
        
        self.training_args = TrainingArguments(
            output_dir=self.calt_config.output_dir,
            num_train_epochs=self.calt_config.num_train_epochs,
            learning_rate=self.calt_config.learning_rate,
            weight_decay=getattr(self.calt_config, 'weight_decay', _DEFAULT_TRAINING_ARGS.weight_decay),
            warmup_ratio=getattr(self.calt_config, 'warmup_ratio', _DEFAULT_TRAINING_ARGS.warmup_ratio),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=test_batch_size,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=getattr(self.calt_config, 'max_grad_norm', _DEFAULT_TRAINING_ARGS.max_grad_norm),
            optim=getattr(self.calt_config, 'optimizer', _DEFAULT_TRAINING_ARGS.optim),
            # Dataloader settings
            dataloader_num_workers=getattr(self.calt_config, 'num_workers', _DEFAULT_TRAINING_ARGS.dataloader_num_workers),
            dataloader_pin_memory=getattr(self.calt_config, 'dataloader_pin_memory', True),
            # Evaluation and saving settings
            eval_strategy=getattr(self.calt_config, 'eval_strategy', 'steps'),
            eval_steps=getattr(self.calt_config, 'eval_steps', 1000),
            save_strategy=getattr(self.calt_config, 'save_strategy', 'steps'),
            save_steps=getattr(self.calt_config, 'save_steps', 1000),
            save_total_limit=getattr(self.calt_config, 'save_total_limit', 1),
            label_names=getattr(self.calt_config, 'label_names', ['labels']),
            save_safetensors=getattr(self.calt_config, 'save_safetensors', False),
            # Logging settings
            logging_strategy=getattr(self.calt_config, 'logging_strategy', 'steps'),
            logging_steps=getattr(self.calt_config, 'logging_steps', 50),
            report_to=report_to,
            # Others
            remove_unused_columns=getattr(self.calt_config, 'remove_unused_columns', False),
            seed=getattr(self.calt_config, 'seed', _DEFAULT_TRAINING_ARGS.seed),
            disable_tqdm=getattr(self.calt_config, 'disable_tqdm', True),
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
            raise ValueError("training_args must be set before building trainer. Call translate_config() first.")
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
