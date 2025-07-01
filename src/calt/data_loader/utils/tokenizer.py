import yaml
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import CharDelimiterSplit
from typing import Optional


def set_tokenizer(
    field: str = "GF",
    max_coeff: int = 100,
    max_degree: int = 10,
    max_length: int = 512,
    vocab_path: Optional[str] = None,
    num_vars: Optional[int] = None,
) -> PreTrainedTokenizerFast:
    """Create or load a tokenizer for polynomial expressions.

    If a vocab_path (YAML file) is provided, it loads a tokenizer from the file.
    Otherwise, it creates a new tokenizer based on the provided parameters.

    Args:
        vocab_path: Optional path to a vocab file (e.g., "vocab.yaml").
        num_vars: Number of variables (x0, x1, ...) in the polynomial. (currently unused)
        field: Field specification ("QQ"/"ZZ" for rational/integer, or
               "GF<p>" for finite field). Used if vocab_path is not provided.
        max_coeff: Maximum absolute value for coefficients in the vocabulary. Used if vocab_path is not provided.
        max_degree: Maximum degree allowed for any variable. Used if vocab_path is not provided.
        max_length: Maximum sequence length the tokenizer will process.

    Returns:
        A pre-configured HuggingFace tokenizer for polynomial expressions.
    """
    if vocab_path:
        with open(vocab_path, "r") as f:
            config = yaml.safe_load(f)
        vocab_list = config["vocab"]
        special_token_map = config["special_vocab"]
        special_tokens = list(special_token_map.values())

    else:
        # Create tokenizer from scratch
        special_tokens = ["[PAD]", "<s>", "</s>", "[CLS]"]
        special_token_map = dict(zip(["pad_token", "bos_token", "eos_token", "cls_token"], special_tokens))

        CONSTS = ["[C]"]
        if field in "ZZ":
            CONSTS += [f"C{i}" for i in range(-max_coeff, max_coeff + 1)]
        elif field.startswith("GF"):
            try:
                p = int(field[2:])
                if p <= 0:
                    raise ValueError()
            except (ValueError, IndexError):
                raise ValueError(f"Invalid field specification for GF(p): {field}")
            CONSTS += [f"C{i}" for i in range(-p + 1, p)]
        else:
            raise ValueError(f"unknown field: {field}")

        ECONSTS = [f"E{i}" for i in range(max_degree + 1)]
        vocab_list = CONSTS + ECONSTS + ["[SEP]"]

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

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, model_max_length=max_length, **special_token_map)
    return tokenizer
