# Lexer and vocabulary

The lexer and vocabulary are configured via `lexer.yaml`; the file format and top-level keys are described in [Configuration](../configuration/#lexeryaml-io-and-vocabulary-iopipeline). Below are the API references for the classes that consume that configuration.

## UnifiedLexer

```
UnifiedLexer(
    vocab_config: VocabConfig,
    number_policy: Optional[NumberPolicy] = None,
    strict: bool = True,
    include_base_vocab: bool = True,
)
```

Bases: `AbstractPreProcessor`

Unified regex-based lexer for tokenizing input strings.

This lexer converts raw input strings into token sequences based on vocabulary configuration. It follows longest-match principle for reserved tokens and supports configurable number tokenization.

Examples:

```
>>> from calt.io.vocabulary import VocabConfig
>>> from calt.io.preprocessor import UnifiedLexer, NumberPolicy
>>>
>>> vocab_config = VocabConfig([], {}).from_config("config/vocab.yaml")
>>> number_policy = NumberPolicy(sign=False, digit_group=1)
>>> lexer = UnifiedLexer(vocab_config, number_policy=number_policy)
>>>
>>> tokens = lexer.tokenize("C-50*x1^2 + 3.14")
>>> # Returns: ["C-50", "*", "x1", "^", "2", "+", "3", ".", "1", "4"]
```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `vocab_config` | `VocabConfig` | Vocabulary configuration. | *required* | | `number_policy` | `Optional[NumberPolicy]` | Policy for tokenizing numbers. If None, uses default. | `None` | | `strict` | `bool` | If True, raise error on unknown characters. If False, emit . | `True` | | `include_base_vocab` | `bool` | Whether to include base vocabulary tokens. | `True` |

Source code in `src/calt/io/preprocessor/lexer.py`

```
def __init__(
    self,
    vocab_config: VocabConfig,
    number_policy: Optional[NumberPolicy] = None,
    strict: bool = True,
    include_base_vocab: bool = True,
):
    """Initialize the unified lexer.

    Args:
        vocab_config: Vocabulary configuration.
        number_policy: Policy for tokenizing numbers. If None, uses default.
        strict: If True, raise error on unknown characters. If False, emit <unk>.
        include_base_vocab: Whether to include base vocabulary tokens.
    """
    # Initialize post-processor base class
    super().__init__()

    self.number_policy = number_policy or NumberPolicy()
    self.strict = strict
    self.include_base_vocab = include_base_vocab

    # Extend vocab_config with required tokens based on number_policy
    self.vocab_config = self._extend_vocab_for_number_policy(vocab_config)

    # Build reserved tokens
    self._build_reserved_tokens()

    # Build regex patterns
    self._build_patterns()
```

## NumberPolicy

```
NumberPolicy(
    sign: bool = False, digit_group: int = 0, allow_float: bool = True
)
```

Policy for tokenizing numbers.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `sign` | `bool` | How to handle sign. True (attach) means sign is part of the number, False (separate) means sign is a separate token. | | `digit_group` | `int` | Group digits. 0 = no split, d>=1 = split every d digits. | | `allow_float` | `bool` | Whether to allow floating point numbers. |

## VocabConfig

```
VocabConfig(
    vocab: list[str],
    special_tokens: dict[str, str],
    include_base_vocab=True,
    include_base_special_tokens=True,
)
```

Source code in `src/calt/io/vocabulary/config.py`

```
def __init__(
    self,
    vocab: list[str],
    special_tokens: dict[str, str],
    include_base_vocab=True,
    include_base_special_tokens=True,
):
    self.vocab = vocab
    self.special_tokens = special_tokens

    if include_base_vocab:
        self.vocab = BASE_VOCAB + self.vocab
    if include_base_special_tokens:
        self.special_tokens = BASE_SPECIAL_TOKENS | self.special_tokens
```
