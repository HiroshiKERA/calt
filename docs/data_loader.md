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

### Tokenizer
Build or load a tokenizer for polynomial expressions and configure the vocabulary.
::: calt.data_loader.utils.tokenizer.set_tokenizer
::: calt.data_loader.utils.tokenizer.VocabConfig

### Visualization utilities (optional)
Quickly render visual diffs between predictions and references.
::: calt.data_loader.utils.comparison_vis.display_with_diff
::: calt.data_loader.utils.comparison_vis.load_eval_results
::: calt.data_loader.utils.comparison_vis.parse_poly
