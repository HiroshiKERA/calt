"""
Loader for GPT-2 model.

Handles conversion from unified config format (cfg.model) to GPT2Config
and creates GPT-2 model instances.
"""

from transformers import GPT2Config

from ..loader import ModelLoader
from .config_mapping import create_gpt2_config
from .model import GPT2ForPromptedGeneration


class GPT2Loader(ModelLoader):
    """Loader for GPT-2 decoder-only model."""

    def translate_config(self) -> GPT2Config:
        """Convert unified config format to GPT2Config."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for GPT-2 model")
        self.config = create_gpt2_config(self.calt_config, self.tokenizer)
        return self.config

    def build_model(self) -> GPT2ForPromptedGeneration:
        """Create GPT-2 model instance."""
        if self.config is None:
            raise ValueError(
                "config must be set before building model. Call translate_config() first."
            )
        return GPT2ForPromptedGeneration(config=self.config)
