from typing import Optional, TypedDict

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from .vocabs.base import VocabConfig

def get_tokenizer(
    vocab_config: VocabConfig,
) -> PreTrainedTokenizerFast:

    vocab_list = vocab_config.vocab
    special_token_map = vocab_config.special_tokens
    special_tokens = list(special_token_map.values())

    vocab = dict(zip(vocab_list, range(len(vocab_list))))

    tok = Tokenizer(WordLevel(vocab))
    tok.pre_tokenizer = CharDelimiterSplit(" ")
    tok.add_special_tokens(special_tokens)
    tok.enable_padding()
    tok.no_truncation()

    bos_token = special_token_map["bos_token"]
    eos_token = special_token_map["eos_token"]
    tok.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tok.token_to_id(bos_token)),
            (eos_token, tok.token_to_id(eos_token)),
        ],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        **special_token_map,
    )
    return tokenizer
