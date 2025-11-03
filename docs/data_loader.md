## Data Loader

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

### Entry point

::: calt.data_loader.data_loader.load_data

- `load_data` accepts either a legacy ``processor_name`` or a fully-configured
  ``processor`` (an :class:`~calt.data_loader.utils.preprocessor.AbstractPreprocessor`
  instance). When both are supplied, the explicit ``processor`` wins, enabling
  advanced pipelines without breaking backward compatibility.

### Dataset and collator
::: calt.data_loader.utils.data_collator.StandardDataset
::: calt.data_loader.utils.data_collator.StandardDataCollator

### Preprocessing (expression → internal tokens)
::: calt.data_loader.utils.preprocessor.AbstractPreprocessor
::: calt.data_loader.utils.preprocessor.ProcessorChain
::: calt.data_loader.utils.preprocessor.CoefficientPostfixProcessor
::: calt.data_loader.utils.preprocessor.PolynomialToInternalProcessor
::: calt.data_loader.utils.preprocessor.IntegerToInternalProcessor

#### Internal token layout
`PolynomialToInternalProcessor` emits coefficient/exponent tokens (`C…` / `E…`) and is the
single implementation used for both polynomial and integer preprocessing
(`num_variables=0`). Key rules:

- **Digit grouping** – an optional `digit_group_size` controls how numeric literals are
  chunked. When the size is `None` or ≤ 0, the entire literal (including any sign and
  leading zeros) is kept in a single coefficient token such as `C-200` or `C007`. When
  the size is positive, digits are split *right-aligned* (least-significant side). The
  sign, if present, is attached to the first chunk only. Examples:

  | Input literal | `digit_group_size=4` | `digit_group_size=3` | `digit_group_size=2` |
  | ------------- | ------------------- | ------------------- | ------------------- |
  | `12345`       | `C1 C2345`           | `C12 C345`           | `C1 C23 C45`         |
  | `-12345`      | `C-1 C2345`          | `C-12 C345`          | `C-1 C23 C45`        |

- **Polynomial terms** (`num_variables > 0`) – every term is emitted as the grouped
  coefficient tokens followed by exactly `num_variables` exponent tokens. For example,
  with two variables and `digit_group_size=3`,
  `12345*x0*x1 -> C12 C345 E1 E1` and `-12345*x0*x1 -> C-12 C345 E1 E1`. A zero
  coefficient is always encoded as a single `C0`.

- **Integer mode** (`num_variables=0`) – the same chunking logic is applied to each
  `|`-separated integer. Encoded parts are joined with `[SEP]`, e.g.
  `-0012 | 34 -> C-0012 [SEP] C34` and, with `digit_group_size=3`,
  `100 | 2000 -> C100 [SEP] C2 C000`. Decoding rejoins the coefficient chunks and restores
  the original separators.

- **Compatibility wrapper** – `IntegerToInternalProcessor` simply delegates to
  `PolynomialToInternalProcessor(num_variables=0, …)` and is kept for legacy callers.

#### Processor chaining example

```python
from calt.data_loader.utils.preprocessor import (
    PolynomialToInternalProcessor,
    CoefficientPostfixProcessor,
    ProcessorChain,
)
from calt.data_loader.data_loader import load_data

poly = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=200)
postfix = CoefficientPostfixProcessor()
chain = ProcessorChain([poly, postfix])

assert chain.encode("123*x0*x1^2") == "E1 E2 C123"

dataset, tokenizer, collator = load_data(
    train_dataset_path="dataset/train.txt",
    test_dataset_path="dataset/test.txt",
    field="ZZ",
    num_variables=2,
    max_degree=3,
    max_coeff=200,
    processor=chain,
)
```

You can append additional custom processors to `ProcessorChain` to insert validation,
data augmentation, or token-level rewrites that should run before batching.

### Tokenizer
Build or load a tokenizer for polynomial expressions and configure the vocabulary.
::: calt.data_loader.utils.tokenizer.set_tokenizer
::: calt.data_loader.utils.tokenizer.VocabConfig

### Visualization utilities (optional)
Quickly render visual diffs between predictions and references.
::: calt.data_loader.utils.comparison_vis.display_with_diff
::: calt.data_loader.utils.comparison_vis.load_eval_results
::: calt.data_loader.utils.comparison_vis.parse_poly

### Changelog

- Added `ProcessorChain` and `CoefficientPostfixProcessor` for composable preprocessing,
  along with the new `processor` argument on `load_data`. Existing ``processor_name``
  flows continue to work unchanged.
