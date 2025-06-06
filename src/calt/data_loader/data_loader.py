"""Data loading utilities for the Transformer Algebra project.

This module defines helper functions that build the training and evaluation
`Dataset`, `Tokenizer`, and `DataCollator` objects used throughout the
library.  In particular, the `data_loader` factory translates symbolic
polynomial expressions into the internal token representation expected by the
Transformer models.
"""

from .utils.data_collator import StandardDataset, StandardDataCollator
from .utils.preprocessor import SymbolicToInternalProcessor, IntegerToInternalProcessor
from .utils.tokenizer import set_tokenizer
from transformers import PreTrainedTokenizerFast as StandardTokenizer
from typing import Tuple


def data_loader(
    train_dataset_path: str,
    test_dataset_path: str,
    field: str,
    num_variables: int,
    max_degree: int,
    max_coeff: int,
    max_length: int = 512,
    processor_name: str = "polynomial",
) -> Tuple[StandardDataset, StandardTokenizer, StandardDataCollator]:
    """Create dataset, tokenizer and data-collator objects.

    Parameters
    ----------
    train_dataset_path : str
        Path to the file that stores the *training* samples.
    test_dataset_path : str
        Path to the file that stores the *evaluation* samples.
    field : str
        Finite-field identifier (e.g. ``"Q"`` for the rationals or ``"Zp"``
        for a prime field) used to generate the vocabulary.
    num_variables : int
        Maximum number of symbolic variables (\(x_1, \dots, x_n\)) that can
        appear in a polynomial.
    max_degree : int
        Maximum total degree allowed for any monomial term.
    max_coeff : int
        Maximum absolute value of the coefficients appearing in the data.
    max_length : int, default ``512``
        Hard upper bound on the token sequence length.  Longer sequences will
        be *right-truncated*.
    processor_name : str, default ``"polynomial"``
        Name of the processor to use for converting symbolic expressions into
        internal token IDs.  The default processor is ``"polynomial"``, which
        handles polynomial expressions.  The alternative processor is
        ``"integer"``, which handles integer expressions.

    Returns
    -------
    Tuple[StandardDataset, StandardTokenizer, StandardDataCollator]
        1. ``dataset``  – a ``dict`` with ``"train"`` and ``"test"`` splits
           containing :class:`StandardDataset` instances.
        2. ``tokenizer`` – a :class:`PreTrainedTokenizerFast` capable of
           encoding symbolic expressions into token IDs and vice versa.
        3. ``data_collator`` – a callable that assembles batches and applies
           dynamic padding so they can be fed to a HuggingFace ``Trainer``.
    """
    if processor_name == "polynomial":
        preprocessor = SymbolicToInternalProcessor(
            num_variables=num_variables,
            max_degree=max_degree,
            max_coeff=max_coeff,
        )
    elif processor_name == "numeric":
        preprocessor = IntegerToInternalProcessor(max_coeff=max_coeff)
    else:
        raise ValueError(f"Unknown processor: {processor_name}")

    train_dataset = StandardDataset(train_dataset_path, preprocessor)
    test_dataset = StandardDataset(test_dataset_path, preprocessor)
    tokenizer = set_tokenizer(
        num_vars=num_variables,
        field=field,
        max_degree=max_degree,
        max_coeff=max_coeff,
        max_length=max_length,
    )
    data_collator = StandardDataCollator(tokenizer)
    dataset = {"train": train_dataset, "test": test_dataset}
    return dataset, tokenizer, data_collator
