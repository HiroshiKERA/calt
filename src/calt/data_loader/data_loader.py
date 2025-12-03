"""Data loading utilities for the Transformer Algebra project.

This module defines helper functions that build the training and evaluation
`Dataset`, `Tokenizer`, and `DataCollator` objects used throughout the
library.  In particular, the `load_data` factory translates symbolic
polynomial expressions into the internal token representation expected by the
Transformer models.
"""

import logging

import yaml
from transformers import PreTrainedTokenizerFast as StandardTokenizer

from .utils.data_collator import (
    StandardDataCollator,
    StandardDataset,
    _read_data_from_file,
)
from .utils.preprocessor import (
    AbstractPreprocessor,
    IntegerToInternalProcessor,
    PolynomialToInternalProcessor,
)
from .utils.tokenizer import VocabConfig, set_tokenizer

logger = logging.getLogger(__name__)


def load_data(
    train_dataset_path: str,
    test_dataset_path: str,
    field: str,
    num_variables: int,
    max_degree: int,
    max_coeff: int,
    max_length: int = 512,
    processor_name: str | None = "polynomial",
    processor: AbstractPreprocessor | None = None,
    vocab_path: str | None = None,
    num_train_samples: int | None = None,
    num_test_samples: int | None = None,
    digit_group_size: int | None = None,
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
        processor_name (str | None, optional):
            Name of the processor to use for converting symbolic expressions into
            internal token IDs. The default processor is ``"polynomial"``, which
            handles polynomial expressions. The alternative processor is
            ``"integer"``, which handles integer expressions. Defaults to
            ``"polynomial"``. Ignored when ``processor`` is supplied.
        processor (AbstractPreprocessor | None, optional):
            Preprocessor instance applied directly to expressions. When provided, it
            takes precedence over ``processor_name`` and allows chaining multiple
            preprocessors together. Defaults to None.
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
    selected_preprocessor = processor

    if selected_preprocessor is None:
        resolved_name = processor_name or "polynomial"
        if resolved_name == "polynomial":
            selected_preprocessor = PolynomialToInternalProcessor(
                num_variables=num_variables,
                max_degree=max_degree,
                max_coeff=max_coeff,
                digit_group_size=digit_group_size,
            )
        elif resolved_name == "integer":
            selected_preprocessor = IntegerToInternalProcessor(
                max_coeff=max_coeff, digit_group_size=digit_group_size
            )
        else:
            raise ValueError(f"Unknown processor: {resolved_name}")

    train_input_texts, train_target_texts = _read_data_from_file(
        train_dataset_path, max_samples=num_train_samples
    )
    train_dataset = StandardDataset(
        input_texts=train_input_texts,
        target_texts=train_target_texts,
        preprocessor=selected_preprocessor,
    )

    test_input_texts, test_target_texts = _read_data_from_file(
        test_dataset_path, max_samples=num_test_samples
    )
    test_dataset = StandardDataset(
        input_texts=test_input_texts,
        target_texts=test_target_texts,
        preprocessor=selected_preprocessor,
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
