## Data Loader

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

### Entry point

::: calt.data_loader.data_loader.load_data

### Dataset and collator
::: calt.data_loader.utils.data_collator.StandardDataset
::: calt.data_loader.utils.data_collator.StandardDataCollator

### Preprocessing (expression → internal tokens)
::: calt.data_loader.utils.preprocessor.PolynomialToInternalProcessor
::: calt.data_loader.utils.preprocessor.IntegerToInternalProcessor

#### Internal token layout
- `PolynomialToInternalProcessor` now accepts an optional `digit_group_size` that controls how coefficients or integer strings are split into `C...` chunks without zero padding. A positive size groups digits from the most-significant side (e.g., `12345` with `digit_group_size=3` becomes `C12 C345` and `-12345` becomes `C-12 C345`), while `None` keeps the legacy one-token-per-coefficient / one-token-per-digit behaviour.
- With `num_variables > 0`, each term is emitted as one or more coefficient chunks followed by exactly `num_variables` exponent tokens: `C...` repeated `N >= 1` times and `E...` repeated `num_variables` times. Coefficient `0` always stays a single `C0` token for compatibility.
- Setting `num_variables=0` switches the processor into integer mode so that a `|`-delimited list of non-negative integers maps to `C...` chunks separated by `[SEP]`, and decoding concatenates the chunks back before restoring the separators.
- `IntegerToInternalProcessor` remains available as a deprecated compatibility wrapper that routes to `PolynomialToInternalProcessor(num_variables=0, ...)`; prefer the unified processor in new code.

### Tokenizer
Build or load a tokenizer for polynomial expressions and configure the vocabulary.
::: calt.data_loader.utils.tokenizer.set_tokenizer
::: calt.data_loader.utils.tokenizer.VocabConfig

### Visualization utilities (optional)
Quickly render visual diffs between predictions and references.
::: calt.data_loader.utils.comparison_vis.display_with_diff
::: calt.data_loader.utils.comparison_vis.load_eval_results
::: calt.data_loader.utils.comparison_vis.parse_poly
