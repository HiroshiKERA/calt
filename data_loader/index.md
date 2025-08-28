## Data Loader

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

### Entry point

Create dataset, tokenizer and data-collator objects.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `train_dataset_path` | `str` | Path to the file that stores the "training" samples. | *required* | | `test_dataset_path` | `str` | Path to the file that stores the "evaluation" samples. | *required* | | `field` | `str` | Finite-field identifier (e.g. "Q" for the rationals or "Zp" for a prime field) used to generate the vocabulary. | *required* | | `num_variables` | `int` | Maximum number of symbolic variables ((x_1, \\dots, x_n)) that can appear in a polynomial. | *required* | | `max_degree` | `int` | Maximum total degree allowed for any monomial term. | *required* | | `max_coeff` | `int` | Maximum absolute value of the coefficients appearing in the data. | *required* | | `max_length` | `int` | Hard upper bound on the token sequence length. Longer sequences will be right-truncated. Defaults to 512. | `512` | | `processor_name` | `str` | Name of the processor to use for converting symbolic expressions into internal token IDs. The default processor is "polynomial", which handles polynomial expressions. The alternative processor is "integer", which handles integer expressions. Defaults to "polynomial". | `'polynomial'` | | `vocab_path` | `str | None` | Path to the vocabulary configuration file. If None, a default vocabulary will be generated based on the field, max_degree, and max_coeff parameters. Defaults to None. | `None` | | `num_train_samples` | `int | None` | Maximum number of training samples to load. If None or -1, all available training samples will be loaded. Defaults to None. | `None` | | `num_test_samples` | `int | None` | Maximum number of test samples to load. If None or -1, all available test samples will be loaded. Defaults to None. | `None` |

Returns:

| Type | Description | | --- | --- | | `tuple[dict[str, StandardDataset], PreTrainedTokenizerFast, StandardDataCollator]` | tuple\[dict[str, StandardDataset], StandardTokenizer, StandardDataCollator\]: 1. dataset - a dict with "train" and "test" splits containing StandardDataset instances. 2. tokenizer - a PreTrainedTokenizerFast capable of encoding symbolic expressions into token IDs and vice versa. 3. data_collator - a callable that assembles batches and applies dynamic padding so they can be fed to a HuggingFace Trainer. |

Source code in `src/calt/data_loader/data_loader.py`

```
def load_data(
    train_dataset_path: str,
    test_dataset_path: str,
    field: str,
    num_variables: int,
    max_degree: int,
    max_coeff: int,
    max_length: int = 512,
    processor_name: str = "polynomial",
    vocab_path: str | None = None,
    num_train_samples: int | None = None,
    num_test_samples: int | None = None,
) -> tuple[dict[str, StandardDataset], StandardTokenizer, StandardDataCollator]:
    """Create dataset, tokenizer and data-collator objects.

    Args:
        train_dataset_path (str):
            Path to the file that stores the "training" samples.
        test_dataset_path (str):
            Path to the file that stores the "evaluation" samples.
        field (str):
            Finite-field identifier (e.g. ``"Q"`` for the rationals or ``"Zp"``
            for a prime field) used to generate the vocabulary.
        num_variables (int):
            Maximum number of symbolic variables (\(x_1, \dots, x_n\)) that can
            appear in a polynomial.
        max_degree (int):
            Maximum total degree allowed for any monomial term.
        max_coeff (int):
            Maximum absolute value of the coefficients appearing in the data.
        max_length (int, optional):
            Hard upper bound on the token sequence length. Longer sequences will
            be right-truncated. Defaults to 512.
        processor_name (str, optional):
            Name of the processor to use for converting symbolic expressions into
            internal token IDs. The default processor is ``"polynomial"``, which
            handles polynomial expressions. The alternative processor is
            ``"integer"``, which handles integer expressions. Defaults to
            ``"polynomial"``.
        vocab_path (str | None, optional):
            Path to the vocabulary configuration file. If None, a default vocabulary
            will be generated based on the field, max_degree, and max_coeff parameters.
            Defaults to None.
        num_train_samples (int | None, optional):
            Maximum number of training samples to load. If None or -1, all available
            training samples will be loaded. Defaults to None.
        num_test_samples (int | None, optional):
            Maximum number of test samples to load. If None or -1, all available
            test samples will be loaded. Defaults to None.

    Returns:
        tuple[dict[str, StandardDataset], StandardTokenizer, StandardDataCollator]:
            1. ``dataset`` - a ``dict`` with ``"train"`` and ``"test"`` splits
               containing ``StandardDataset`` instances.
            2. ``tokenizer`` - a ``PreTrainedTokenizerFast`` capable of encoding
               symbolic expressions into token IDs and vice versa.
            3. ``data_collator`` - a callable that assembles batches and applies
               dynamic padding so they can be fed to a HuggingFace ``Trainer``.
    """
    if processor_name == "polynomial":
        preprocessor = PolynomialToInternalProcessor(
            num_variables=num_variables,
            max_degree=max_degree,
            max_coeff=max_coeff,
        )
    elif processor_name == "integer":
        preprocessor = IntegerToInternalProcessor(max_coeff=max_coeff)
    else:
        raise ValueError(f"Unknown processor: {processor_name}")

    train_input_texts, train_target_texts = _read_data_from_file(
        train_dataset_path, max_samples=num_train_samples
    )
    train_dataset = StandardDataset(
        input_texts=train_input_texts,
        target_texts=train_target_texts,
        preprocessor=preprocessor,
    )

    test_input_texts, test_target_texts = _read_data_from_file(
        test_dataset_path, max_samples=num_test_samples
    )
    test_dataset = StandardDataset(
        input_texts=test_input_texts,
        target_texts=test_target_texts,
        preprocessor=preprocessor,
    )

    vocab_config: VocabConfig | None = None
    if vocab_path:
        with open(vocab_path, "r") as f:
            vocab_config = yaml.safe_load(f)

    tokenizer = set_tokenizer(
        field=field,
        max_degree=max_degree,
        max_coeff=max_coeff,
        max_length=max_length,
        vocab_config=vocab_config,
    )
    data_collator = StandardDataCollator(tokenizer)
    dataset = {"train": train_dataset, "test": test_dataset}
    return dataset, tokenizer, data_collator

```

### Dataset and collator

Bases: `Dataset`

Source code in `src/calt/data_loader/utils/data_collator.py`

```
def __init__(
    self,
    input_texts: list[str],
    target_texts: list[str],
    preprocessor: AbstractPreprocessor,
    **extra_fields,
) -> None:
    self.input_texts = input_texts
    self.target_texts = target_texts
    self.preprocessor = preprocessor
    self.extra_fields = extra_fields

    num_samples = len(self.input_texts)
    if len(self.target_texts) != num_samples:
        raise ValueError(
            "input_texts and target_texts must have the same number of samples."
        )

    for name, data in self.extra_fields.items():
        if len(data) != num_samples:
            raise ValueError(
                f"Extra field '{name}' has {len(data)} samples, but {num_samples} were expected."
            )

```

## load_file

```
load_file(
    data_path: str,
    preprocessor: AbstractPreprocessor,
    max_samples: int | None = None,
) -> StandardDataset

```

Load data from a file and create a `StandardDataset` instance.

This method maintains backward compatibility with the previous file-based initialization.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `data_path` | `str` | Path to the data file. | *required* | | `preprocessor` | `AbstractPreprocessor` | Preprocessor instance. | *required* | | `max_samples` | `int | None` | Maximum number of samples to load. Use -1 or None to load all samples. Defaults to None. | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `StandardDataset` | `StandardDataset` | Loaded dataset instance. |

Source code in `src/calt/data_loader/utils/data_collator.py`

```
@classmethod
def load_file(
    cls,
    data_path: str,
    preprocessor: AbstractPreprocessor,
    max_samples: int | None = None,
) -> "StandardDataset":
    """Load data from a file and create a ``StandardDataset`` instance.

    This method maintains backward compatibility with the previous file-based initialization.

    Args:
        data_path (str): Path to the data file.
        preprocessor (AbstractPreprocessor): Preprocessor instance.
        max_samples (int | None, optional): Maximum number of samples to load.
            Use -1 or None to load all samples. Defaults to None.

    Returns:
        StandardDataset: Loaded dataset instance.
    """
    input_texts, target_texts = _read_data_from_file(data_path, max_samples)
    return cls(
        input_texts=input_texts,
        target_texts=target_texts,
        preprocessor=preprocessor,
    )

```

## __getitem__

```
__getitem__(idx: int) -> dict[str, str]

```

Get dataset item and convert to internal representation.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `idx` | `int` | Index of the item to retrieve. | *required* |

Returns:

| Type | Description | | --- | --- | | `dict[str, str]` | dict\[str, str\]: A mapping with keys "input" and "target". |

Source code in `src/calt/data_loader/utils/data_collator.py`

```
def __getitem__(self, idx: int) -> dict[str, str]:
    """Get dataset item and convert to internal representation.

    Args:
        idx (int): Index of the item to retrieve.

    Returns:
        dict[str, str]: A mapping with keys ``"input"`` and ``"target"``.
    """
    src = self.preprocessor(self.input_texts[idx])
    tgt = self.preprocessor(self.target_texts[idx])
    return {"input": src, "target": tgt}

```

Source code in `src/calt/data_loader/utils/data_collator.py`

```
def __init__(self, tokenizer: Tokenizer = None) -> None:
    self.tokenizer = tokenizer

```

## __call__

```
__call__(batch)

```

Collate a batch of data samples.

If a tokenizer is provided, it tokenizes `input` and `target` attributes. Other attributes starting with `target_` are prefixed with `decoder_` and padded.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `batch` | `list[dict[str, Any]]` | Mini-batch samples. | *required* |

Returns:

| Type | Description | | --- | --- | | | dict\[str, torch.Tensor | list[str]\]: Batched tensors and/or lists. |

Source code in `src/calt/data_loader/utils/data_collator.py`

```
def __call__(self, batch):
    """Collate a batch of data samples.

    If a tokenizer is provided, it tokenizes ``input`` and ``target`` attributes.
    Other attributes starting with ``target_`` are prefixed with ``decoder_`` and padded.

    Args:
        batch (list[dict[str, Any]]): Mini-batch samples.

    Returns:
        dict[str, torch.Tensor | list[str]]: Batched tensors and/or lists.
    """
    batch_dict = {}

    # Get the attributes from the first item in the batch.
    attributes = batch[0].keys()

    if self.tokenizer is None:
        # If no tokenizer is provided, return the batch as is.
        for attribute in attributes:
            attribute_batch = [item[attribute] for item in batch]
            batch_dict[attribute] = attribute_batch

        return batch_dict

    for attribute in attributes:
        attribute_batch = [item[attribute] for item in batch]

        if attribute == "input":
            # Tokenize the input sequences.
            inputs = self.tokenizer(
                attribute_batch, padding="longest", return_tensors="pt"
            )
            batch_dict["input_ids"] = inputs["input_ids"]
            batch_dict["attention_mask"] = inputs["attention_mask"]

        elif attribute == "target":
            # Tokenize the target sequences.
            targets = self.tokenizer(
                attribute_batch, padding="longest", return_tensors="pt"
            )
            # Prepare decoder input ids (remove the last token, usually EOS).
            batch_dict["decoder_input_ids"] = targets["input_ids"][
                :, :-1
            ].contiguous()
            # Prepare decoder attention mask accordingly.
            batch_dict["decoder_attention_mask"] = targets["attention_mask"][
                :, :-1
            ].contiguous()

            # Prepare labels for the loss calculation (shift by one, usually remove BOS).
            labels = targets["input_ids"][:, 1:].contiguous()
            label_attention_mask = (
                targets["attention_mask"][:, 1:].contiguous().bool()
            )
            # Set padding tokens in labels to -100 to be ignored by the loss function.
            labels[~label_attention_mask] = -100
            batch_dict["labels"] = labels

        else:
            # For other attributes, if they start with 'target_',
            # prefix them with 'decoder_' (e.g., 'target_aux' becomes 'decoder_aux').
            if attribute.startswith("target_"):
                attribute_key = (
                    "decoder_" + attribute[7:]
                )  #  Corrected key for batch_dict
            else:
                attribute_key = (
                    attribute  # Use original attribute name if no prefix
                )
            # Pad the sequences for these attributes.
            batch_dict[attribute_key] = self._pad_sequences(
                attribute_batch, padding_value=0
            )

    return batch_dict

```

### Preprocessing (expression → internal tokens)

Bases: `AbstractPreprocessor`

Convert SageMath-style expressions to/from internal token representation.

Example (to_internal): "2*x1^2*x0 + 5\*x0 - 3" -> "C2 E1 E2 C5 E1 E0 C-3 E0 E0" (for `num_vars=2`)

Example (to_original): "C2 E2 E1 C5 E1 E0 C-3 E0 E0" -> "2*x0^2*x1 + 5\*x0 - 3"

The internal representation uses

- `C{n}` tokens for coefficients (e.g., `C2`, `C-3`)
- `E{n}` tokens for exponents (e.g., `E1`, `E2`, `E0`)

Each term is represented as a coefficient token followed by exponent tokens for each variable.

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def __init__(self, num_variables: int, max_degree: int, max_coeff: int):
    """Initialize preprocessor parameters.

    Args:
        num_variables (int): Number of variables in the polynomial (e.g., x0, x1, ...).
        max_degree (int): Maximum degree of the polynomial.
        max_coeff (int): Maximum coefficient value in the polynomial.
    """
    if num_variables < 0:
        raise ValueError("num_variables must be positive")
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    if max_coeff <= 0:
        raise ValueError("max_coeff must be positive")

    self.num_variables = num_variables
    self.max_degree = max_degree
    self.max_coeff = max_coeff
    self.var_name_to_index = {f"x{i}": i for i in range(num_variables)}

```

## encode

```
encode(text: str) -> str

```

Process a symbolic text into internal token representation.

If the text contains the '|' separator character, each part is processed separately and joined with '[SEP]' token.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `text` | `str` | Input symbolic text to process. | *required* |

Returns:

| Name | Type | Description | | --- | --- | --- | | `str` | `str` | String in the internal token representation. |

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def encode(self, text: str) -> str:
    """Process a symbolic text into internal token representation.

    If the text contains the '|' separator character, each part is processed
    separately and joined with '[SEP]' token.

    Args:
        text (str): Input symbolic text to process.

    Returns:
        str: String in the internal token representation.
    """
    # If text contains '|', process each part separately and join with [SEP]
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
        internals = [self._poly_to_encode(p) for p in parts]
        processed_string = " [SEP] ".join(internals)
    else:
        processed_string = self._poly_to_encode(text)

    return processed_string

```

## decode

```
decode(tokens: str) -> str

```

Convert an internal token string back to a symbolic polynomial expression.

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def decode(self, tokens: str) -> str:
    """Convert an internal token string back to a symbolic polynomial expression."""
    if "[SEP]" in tokens:
        parts = tokens.split("[SEP]")
        original_parts = [self._internal_to_poly(p.strip()) for p in parts]
        return " | ".join(original_parts)
    else:
        return self._internal_to_poly(tokens)

```

Bases: `AbstractPreprocessor`

Convert an integer string to/from internal token representation.

Input format examples (to_internal):

- "12345"
- "123|45|678" Output format examples (from_internal):
- "C1 C2 C3 C4 C5"
- "C1 C2 C3 [SEP] C4 C5 [SEP] C6 C7 C8"

The internal representation uses `C{n}` tokens for digits. Parts separated by '|' are converted individually and joined by `[SEP]`. Note: `num_variables`, `max_degree`, `max_coeff` are inherited but not directly used.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `max_coeff` | `int` | The maximum digit value (typically 9). Passed to superclass but primarily used for validation context. | `9` |

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def __init__(self, max_coeff: int = 9):
    """Initialize the processor.

    Args:
        max_coeff (int): The maximum digit value (typically 9). Passed to superclass but primarily used for validation context.
    """
    # Use dummy values for num_variables and max_degree as they are not relevant
    super().__init__(num_variables=0, max_degree=0, max_coeff=max_coeff)

```

## encode

```
encode(text: str) -> str

```

Process an integer string into internal token representation.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `text` | `str` | Input string representing one or more integers separated by '|'. | *required* |

Returns:

| Name | Type | Description | | --- | --- | --- | | `str` | `str` | Internal token representation (e.g., "C1 C2 [SEP] C3 C4"), or "[ERROR_FORMAT]" if any part is not a valid integer string. |

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def encode(self, text: str) -> str:
    """Process an integer string into internal token representation.

    Args:
        text (str): Input string representing one or more integers separated by '|'.

    Returns:
        str: Internal token representation (e.g., "C1 C2 [SEP] C3 C4"), or "[ERROR_FORMAT]" if any part is not a valid integer string.
    """
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
        tokenized_parts = []
        for part in parts:
            tokens = self._number_to_tokens(part)
            if tokens == "[ERROR_FORMAT]":
                # If any part fails, return error for the whole input
                return "[ERROR_FORMAT]"
            tokenized_parts.append(tokens)
        # Join the tokenized parts with [SEP]
        return " [SEP] ".join(tokenized_parts)
    else:
        # If no separator, process the whole string
        return self._number_to_tokens(text.strip())

```

## decode

```
decode(tokens: str) -> str

```

Convert an internal token representation back to an integer string.

Source code in `src/calt/data_loader/utils/preprocessor.py`

```
def decode(self, tokens: str) -> str:
    """Convert an internal token representation back to an integer string."""
    if "[SEP]" in tokens:
        parts = tokens.split("[SEP]")
        # Process each part and join with '|'
        number_parts = [self._tokens_to_number(p.strip()) for p in parts]
        return "|".join(number_parts)
    else:
        # Process the whole string if no separator
        return self._tokens_to_number(tokens.strip())

```

### Tokenizer

Build or load a tokenizer for polynomial expressions and configure the vocabulary.

Create or load a tokenizer for polynomial expressions.

If a `vocab_config` is provided, it builds a tokenizer from the config. Otherwise, it creates a new tokenizer based on the provided parameters.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `field` | `str` | Field specification ("QQ"/"ZZ" for rational/integer, or "GF" for finite field). Used if vocab_config is not provided. | `'GF'` | | `max_coeff` | `int` | Maximum absolute value for coefficients. Used if vocab_config is not provided. | `100` | | `max_degree` | `int` | Maximum degree for any variable. Used if vocab_config is not provided. | `10` | | `max_length` | `int` | Maximum sequence length the tokenizer will process. | `512` | | `vocab_config` | `Optional[VocabConfig]` | Optional dictionary with "vocab" and "special_vocab". | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `PreTrainedTokenizerFast` | `PreTrainedTokenizerFast` | A pre-configured HuggingFace tokenizer for polynomial expressions. |

Source code in `src/calt/data_loader/utils/tokenizer.py`

```
def set_tokenizer(
    field: str = "GF",
    max_coeff: int = 100,
    max_degree: int = 10,
    max_length: int = 512,
    vocab_config: Optional[VocabConfig] = None,
) -> PreTrainedTokenizerFast:
    """Create or load a tokenizer for polynomial expressions.

    If a ``vocab_config`` is provided, it builds a tokenizer from the config.
    Otherwise, it creates a new tokenizer based on the provided parameters.

    Args:
        field (str): Field specification ("QQ"/"ZZ" for rational/integer, or "GF<p>"
            for finite field). Used if ``vocab_config`` is not provided.
        max_coeff (int): Maximum absolute value for coefficients. Used if
            ``vocab_config`` is not provided.
        max_degree (int): Maximum degree for any variable. Used if ``vocab_config`` is
            not provided.
        max_length (int): Maximum sequence length the tokenizer will process.
        vocab_config (Optional[VocabConfig]): Optional dictionary with "vocab" and "special_vocab".

    Returns:
        PreTrainedTokenizerFast: A pre-configured HuggingFace tokenizer for polynomial expressions.
    """
    if vocab_config:
        vocab_list = vocab_config["vocab"]
        special_token_map = vocab_config["special_vocab"]
        special_tokens = list(special_token_map.values())

    else:
        # Create tokenizer from scratch
        special_tokens = ["[PAD]", "<s>", "</s>", "[CLS]"]
        special_token_map = dict(
            zip(
                ["pad_token", "bos_token", "eos_token", "cls_token"],
                special_tokens,
            )
        )

        CONSTS = ["[C]"]
        if field in "ZZ":
            CONSTS += [f"C{i}" for i in range(-max_coeff, max_coeff + 1)]
        elif field.startswith("GF"):
            try:
                p = int(field[2:])
                if p <= 0:
                    raise ValueError()
            except (ValueError, IndexError):
                msg = f"Invalid field specification for GF(p): {field}"
                raise ValueError(msg)
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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        model_max_length=max_length,
        **special_token_map,
    )
    return tokenizer

```

Bases: `TypedDict`

### Visualization utilities (optional)

Quickly render visual diffs between predictions and references.

Render "gold" vs. "pred" with strikethrough on mistakes in "pred".

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `gold` | `Expr | str` | Ground-truth expression. If a string, it will be parsed as a token sequence (e.g., "C1 E1 E1 C-3 E0 E7") via parse_poly. | *required* | | `pred` | `Expr | str` | Model-predicted expression. If a string, it will be parsed as a token sequence via parse_poly. | *required* | | `var_order` | `Sequence[Symbol] | None` | Variable ordering (important for >2 variables). Inferred if None. Also passed to parse_poly if inputs are strings. Defaults to None. | `None` |

Source code in `src/calt/data_loader/utils/comparison_vis.py`

```
def display_with_diff(
    gold: Expr | str,
    pred: Expr | str,
    var_order: Sequence[Symbol] | None = None,
) -> None:
    """Render "gold" vs. "pred" with strikethrough on mistakes in "pred".

    Args:
        gold (sympy.Expr | str):
            Ground-truth expression. If a string, it will be parsed as a token
            sequence (e.g., "C1 E1 E1 C-3 E0 E7") via ``parse_poly``.
        pred (sympy.Expr | str):
            Model-predicted expression. If a string, it will be parsed as a token
            sequence via ``parse_poly``.
        var_order (Sequence[sympy.Symbol] | None, optional):
            Variable ordering (important for >2 variables). Inferred if None. Also
            passed to ``parse_poly`` if inputs are strings. Defaults to None.
    """

    # --- input conversion ------------------------------------------------- #
    if isinstance(gold, str):
        gold = parse_poly(gold, var_names=var_order)
    if isinstance(pred, str):
        pred = parse_poly(pred, var_names=var_order)

    # --- normalize -------------------------------------------------------- #
    if var_order is None:
        var_order = sorted(
            gold.free_symbols.union(pred.free_symbols), key=lambda s: s.name
        )
    gold_poly = Poly(gold.expand(), *var_order)
    pred_poly = Poly(pred.expand(), *var_order)

    gdict = _poly_to_dict(gold_poly)
    pdict = _poly_to_dict(pred_poly)

    # --- diff detection --------------------------------------------------- #
    diff: dict[tuple[int, ...], str] = {}
    for exps in set(gdict) | set(pdict):
        gcoeff = gdict.get(exps, 0)
        pcoeff = pdict.get(exps, 0)
        if pcoeff == 0 and gcoeff != 0:
            continue  # missing term (not highlighted)
        if gcoeff == 0 and pcoeff != 0:
            diff[exps] = "extra"
        elif gcoeff != pcoeff:
            diff[exps] = "coeff_wrong"

    # --- render ----------------------------------------------------------- #
    gold_tex = latex(gold.expand())
    pred_tex = _build_poly_latex(pdict, var_order, diff)

    display(
        Math(
            r"""\begin{aligned}
        \text{Ground truth\,:}\; & {}"""
            + gold_tex
            + r"""\\
        \text{Prediction\,:}\;   & {}"""
            + pred_tex
            + r"""
        \end{aligned}"""
        )
    )

```

Load evaluation results from a JSON file.

The JSON file should contain a list of objects with "generated" and "reference" keys.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str` | Path to the JSON file. | *required* |

Returns:

| Type | Description | | --- | --- | | `tuple[list[str], list[str]]` | tuple\[list[str], list[str]\]: A tuple containing two lists: - List of generated texts. - List of reference texts. |

Source code in `src/calt/data_loader/utils/comparison_vis.py`

```
def load_eval_results(file_path: str) -> tuple[list[str], list[str]]:
    """Load evaluation results from a JSON file.

    The JSON file should contain a list of objects with "generated" and "reference" keys.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - List of generated texts.
            - List of reference texts.
    """
    generated_texts = []
    reference_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        generated_texts.append(item.get("generated", ""))
        reference_texts.append(item.get("reference", ""))

    return generated_texts, reference_texts

```

Convert a math expression string or token sequence to a SymPy polynomial.

This function handles:

1. Standard mathematical notation (e.g., "4*x0 + 4*x1").
1. SageMath-style power notation (e.g., "3*x0^2 + 3*x0").
1. Internal token format (e.g., "C4 E1 E0 C4 E0 E1").

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `text` | `str` | The mathematical expression or token sequence to parse. | *required* | | `var_names` | `Sequence[str | Symbol] | None` | Variable names. Primarily used for the token sequence format to ensure the correct number of variables. For expression strings, variables are inferred, but providing them can ensure they are treated as symbols. | `None` |

Returns:

| Type | Description | | --- | --- | | `Expr` | sympy.Expr: A SymPy expression for the polynomial. |

Source code in `src/calt/data_loader/utils/comparison_vis.py`

```
def parse_poly(text: str, var_names: Sequence[str | Symbol] | None = None) -> Expr:
    """Convert a math expression string or token sequence to a SymPy polynomial.

    This function handles:
    1. Standard mathematical notation (e.g., "4*x0 + 4*x1").
    2. SageMath-style power notation (e.g., "3*x0^2 + 3*x0").
    3. Internal token format (e.g., "C4 E1 E0 C4 E0 E1").

    Args:
        text (str):
            The mathematical expression or token sequence to parse.
        var_names (Sequence[str | sympy.Symbol] | None, optional):
            Variable names. Primarily used for the token sequence format to ensure
            the correct number of variables. For expression strings, variables are
            inferred, but providing them can ensure they are treated as symbols.

    Returns:
        sympy.Expr: A SymPy expression for the polynomial.
    """
    text = text.strip()

    # Heuristic: if the text starts with a 'C' token, attempt to parse it
    # using the token-based parser first.
    if text.startswith("C"):
        try:
            return _parse_poly_from_tokens(text, var_names)
        except (ValueError, IndexError):
            # Fallback to standard expression parsing if token parsing fails.
            # This allows parsing expressions that happen to start with 'C'
            # (e.g., if 'C' is a variable name).
            pass

    # Standard expression parsing
    # Replace SageMath-style power operator '^' with SymPy's '**'
    text_sympy = text.replace("^", "**")

    # Prepare a local dictionary of symbols if var_names are provided
    local_dict = {}
    if var_names:
        if all(isinstance(v, Symbol) for v in var_names):
            symbols_map = {s.name: s for s in var_names}
        else:
            symbols_map = {str(name): Symbol(str(name)) for name in var_names}
        local_dict.update(symbols_map)

    return parse_expr(text_sympy, local_dict=local_dict, evaluate=True)

```
