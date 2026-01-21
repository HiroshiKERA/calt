"""Dataclasses and types for checkpoint save/load."""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from calt.io.preprocessor import UnifiedLexer
from calt.io.vocabulary.config import VocabConfig


@dataclass
class RunBundle:
    """Bundle for full reproduction: model, tokenizer, vocab, and lexer.

    Returned by load_run() when raw_text input format is needed.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    vocab_config: dict | DictConfig
    lexer_config: dict | DictConfig
    vocab: VocabConfig
    lexer: UnifiedLexer
    train_config: dict | DictConfig | None = None
    manifest: dict[str, Any] | None = None
