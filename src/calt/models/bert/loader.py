"""
Loader for BERT model.

Handles conversion from unified config format (cfg.model) to BertConfig
and creates BERT model instances.
"""

from transformers import BertConfig

from ..loader import ModelLoader
from .config_mapping import create_bert_config
from .model import BertForSingleTokenClassification


class BertLoader(ModelLoader):
    """Loader for BERT encoder-only classification model."""

    def translate_config(self) -> BertConfig:
        """Convert unified config format to BertConfig."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for BERT model")
        self.config = create_bert_config(self.calt_config, self.tokenizer)
        return self.config

    def build_model(self) -> BertForSingleTokenClassification:
        """Create BERT model instance."""
        if self.config is None:
            raise ValueError(
                "config must be set before building model. Call translate_config() first."
            )
        return BertForSingleTokenClassification(config=self.config)
