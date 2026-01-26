# Dataset Generator

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and solutions, supports batch writing, and computes incremental statistics.

## High-level API

### Generation flow
::: calt.dataset.sagemath.dataset_generator.DatasetGenerator

### Writing and statistics
::: calt.dataset.sagemath.utils.dataset_writer.DatasetWriter
::: calt.dataset.sagemath.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
::: calt.dataset.sagemath.utils.polynomial_sampler.PolynomialSampler

## SymPy backend

### Generation flow
::: calt.dataset.sympy.dataset_generator.DatasetGenerator

### Writing and statistics
::: calt.dataset.sympy.utils.dataset_writer.DatasetWriter
::: calt.dataset.sympy.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
::: calt.dataset.sympy.utils.polynomial_sampler.PolynomialSampler
::: calt.dataset.sympy.utils.single_polynomial_sampler.SinglePolynomialSampler

## Configuration via `data.yaml`

Example tasks under `calt/examples/*` use a small configuration file `configs/data.yaml`
to drive dataset generation through :class:`calt.dataset.pipeline.DatasetPipeline`.
The typical usage pattern is:

```python
from omegaconf import OmegaConf
from calt.dataset import DatasetPipeline

cfg = OmegaConf.load("configs/data.yaml")
pipeline = DatasetPipeline.from_config(
    cfg.dataset,
    problem_generator=my_problem_generator,
    statistics_calculator=None,
)
pipeline.run()
```

The `dataset` block in `data.yaml` controls all dataset-generation behaviour:

- **`save_dir`**: base directory where all splits (train/test/…) are written.
- **`num_train_samples`**: number of training samples to generate (size of the `"train"` split).
- **`num_test_samples`**: number of test samples to generate (size of the `"test"` split).
- **`batch_size`**: batch size passed to `DatasetGenerator.run` for efficient multiprocessing.
- **`n_jobs`**: number of worker processes used by the generator (`backend="multiprocessing"`).
- **`root_seed`**: global seed used to derive per-sample seeds in the backend.
- **`verbose`**: whether the generator prints progress information.
- **`backend`**: which implementation to use:
  - `"sagemath"` → `calt.dataset.sagemath.dataset_generator.DatasetGenerator`
  - `"sympy"` → `calt.dataset.sympy.dataset_generator.DatasetGenerator`
- **`save_text`**: whether to write human-readable `.txt` files (`text_raw.txt`, etc.).
- **`save_json`**: whether to write `.jsonl` files preserving the nested Python structure.

Under the hood, :class:`DatasetPipeline` resolves the appropriate `DatasetGenerator`
and `DatasetWriter` using `backend`, and then forwards all configuration options when
creating the generator and writer instances.

