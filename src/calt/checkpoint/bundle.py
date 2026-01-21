from dataclasses import dataclass
from omegaconf import DictConfig
from calt.io.vocabulary import BuiltVocab
from calt.io.preprocessor import RegexLexer
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from torch.utils.data import Dataset
from calt.data.collator import StandardDataCollator

@dataclass
class RunBundle:
    cfg: DictConfig              # train.yaml merged/normalized
    vocab_cfg: DictConfig        # vocab.yaml
    vocab: BuiltVocab
    lexer: RegexLexer
    tokenizer: PreTrainedTokenizerFast
    collator: StandardDataCollator
    model: PreTrainedModel
    # optional:
    train_dataset: Dataset | None
    test_dataset: Dataset | None
