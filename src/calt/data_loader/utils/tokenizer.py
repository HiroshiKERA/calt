from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import CharDelimiterSplit
from typing import Optional

SPECIAL_TOKENS = ["[PAD]", "<s>", "</s>", "[CLS]"]
# Create mapping from token names to special token values
SPECIAL_TOKEN_MAP = dict(zip(["pad_token", "bos_token", "eos_token", "cls_token"], SPECIAL_TOKENS))


def set_tokenizer(
    field: str = "GF",
    max_coeff: int = 100,
    max_degree: int = 10,
    max_length: int = 512,
    tokenizer_path: Optional[str] = None,
    num_vars: Optional[int] = None,
) -> PreTrainedTokenizerFast:
    """Create or load a tokenizer for polynomial expressions.

    If a tokenizer_path is provided, it loads a tokenizer from the file.
    Otherwise, it creates a new tokenizer based on the provided parameters.

    Args:
        tokenizer_path: Optional path to a tokenizer file (e.g., "tokenizer.json").
        num_vars: Number of variables (x0, x1, ...) in the polynomial. (currently unused)
        field: Field specification ("QQ"/"ZZ" for rational/integer, or
               "GF<p>" for finite field).
        max_coeff: Maximum absolute value for coefficients in the vocabulary.
        max_degree: Maximum degree allowed for any variable.
        max_length: Maximum sequence length the tokenizer will process.

    Returns:
        A pre-configured HuggingFace tokenizer for polynomial expressions.
    """
    if tokenizer_path:
        # tok = Tokenizer.from_file(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Create tokenizer from scratch
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
        vocab = CONSTS + ECONSTS + ["[SEP]"]
        vocab = dict(zip(vocab, range(len(vocab))))

        tok = Tokenizer(WordLevel(vocab))
        tok.pre_tokenizer = CharDelimiterSplit(" ")
        tok.add_special_tokens(SPECIAL_TOKENS)
        tok.enable_padding()
        tok.no_truncation()

        bos_token = SPECIAL_TOKEN_MAP["bos_token"]
        eos_token = SPECIAL_TOKEN_MAP["eos_token"]
        tok.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, tok.token_to_id(bos_token)),
                (eos_token, tok.token_to_id(eos_token)),
            ],
        )

        # Wrap with HuggingFace's fast tokenizer interface
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, model_max_length=max_length, **SPECIAL_TOKEN_MAP)
    return tokenizer
