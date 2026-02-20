# Overview

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and answers, supports batch writing, and computes incremental statistics.

[DatasetPipeline](#calt.dataset.pipeline.DatasetPipeline) and [DatasetWriter](#datasetwriter) are shared regardless of backend (`sagemath` or `sympy`). For details on each component, see:

- [DatasetWriter](#datasetwriter) — writing samples to disk
- [SageMath backend](../dataset_sagemath/) — `DatasetGenerator` and `PolynomialSampler` for SageMath
- [SymPy backend](../dataset_sympy/) — `DatasetGenerator` and `PolynomialSampler` for SymPy

## DatasetPipeline

```
DatasetPipeline(
    instance_generator,
    statistics_calculator,
    save_dir: str,
    save_text: bool,
    save_json: bool,
    num_train_samples: int,
    num_test_samples: int,
    batch_size: int,
    n_jobs: int,
    root_seed: int,
    verbose: bool,
    backend: str = "sagemath",
)
```

Pipeline for generating train/test datasets with a configurable backend.

Uses an instance generator and optional statistics calculator to produce batches, then writes them to disk via the backend's DatasetWriter. Typically constructed via from_config() with a DictConfig (e.g. from YAML).

Examples:

```
>>> from omegaconf import OmegaConf
>>> from calt.dataset import DatasetPipeline
>>> cfg = OmegaConf.load("configs/dataset.yaml")
>>> pipeline = DatasetPipeline.from_config(
...     cfg.dataset,
...     instance_generator=my_instance_generator,
...     statistics_calculator=my_stats_fn,
... )
>>> pipeline.run()
```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `instance_generator` | | Callable that takes a single integer seed and returns (problem, answer). Used to generate each sample. | *required* | | `statistics_calculator` | | Optional callable(problem, answer) returning a dict of per-sample statistics (e.g. {"problem": {...}, "answer": {...}}). Pass None to skip statistics. | *required* | | `save_dir` | `str` | Directory path to write dataset files. | *required* | | `save_text` | `bool` | Whether to save samples as text files. | *required* | | `save_json` | `bool` | Whether to save metadata (e.g. statistics) as JSON. | *required* | | `num_train_samples` | `int` | Number of training samples to generate. | *required* | | `num_test_samples` | `int` | Number of test samples to generate. | *required* | | `batch_size` | `int` | Number of samples per batch during generation. | *required* | | `n_jobs` | `int` | Number of parallel jobs for the backend generator. | *required* | | `root_seed` | `int` | Base seed for reproducibility; job seeds are derived from this. | *required* | | `verbose` | `bool` | Whether to print progress. | *required* | | `backend` | `str` | Backend name for generation and writing ("sagemath" or "sympy"). | `'sagemath'` |

Source code in `src/calt/dataset/pipeline.py`

```
def __init__(
    self,
    instance_generator,
    statistics_calculator,
    save_dir: str,
    save_text: bool,
    save_json: bool,
    num_train_samples: int,
    num_test_samples: int,
    batch_size: int,
    n_jobs: int,
    root_seed: int,
    verbose: bool,
    backend: str = "sagemath",
) -> None:
    """Initialize the dataset pipeline.

    Args:
        instance_generator: Callable that takes a single integer seed and
            returns (problem, answer). Used to generate each sample.
        statistics_calculator: Optional callable(problem, answer) returning
            a dict of per-sample statistics (e.g. {"problem": {...}, "answer": {...}}).
            Pass None to skip statistics.
        save_dir: Directory path to write dataset files.
        save_text: Whether to save samples as text files.
        save_json: Whether to save metadata (e.g. statistics) as JSON.
        num_train_samples: Number of training samples to generate.
        num_test_samples: Number of test samples to generate.
        batch_size: Number of samples per batch during generation.
        n_jobs: Number of parallel jobs for the backend generator.
        root_seed: Base seed for reproducibility; job seeds are derived from this.
        verbose: Whether to print progress.
        backend: Backend name for generation and writing ("sagemath" or "sympy").
    """
    self.instance_generator = instance_generator
    self.statistics_calculator = statistics_calculator
    self.save_dir = save_dir
    self.save_text = save_text
    self.save_json = save_json
    self.num_train_samples = num_train_samples
    self.num_test_samples = num_test_samples
    self.batch_size = batch_size
    self.n_jobs = n_jobs
    self.root_seed = root_seed
    self.verbose = verbose
    self.backend = backend
```

### from_config

```
from_config(
    config: DictConfig, instance_generator, statistics_calculator=None
) -> "DatasetPipeline"
```

Build a DatasetPipeline from a DictConfig.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `config` | `DictConfig` | DictConfig for the dataset. Expected keys: save_dir, num_train_samples, num_test_samples, batch_size, n_jobs, root_seed. Optional (with defaults): save_text=True, save_json=True, verbose=True, backend="sagemath". Missing required keys will raise when building the pipeline. | *required* | | `instance_generator` | | Callable(seed) -> (problem, answer). Required. | *required* | | `statistics_calculator` | | Optional callable(problem, answer) -> dict. Defaults to None (no per-sample statistics). | `None` |

Returns:

| Type | Description | | --- | --- | | `'DatasetPipeline'` | DatasetPipeline instance configured with config and the given callables. |

Source code in `src/calt/dataset/pipeline.py`

```
@classmethod
def from_config(
    cls,
    config: DictConfig,
    instance_generator,
    statistics_calculator=None,
) -> "DatasetPipeline":
    """Build a DatasetPipeline from a DictConfig.

    Args:
        config: DictConfig for the dataset. Expected keys: save_dir,
            num_train_samples, num_test_samples, batch_size, n_jobs, root_seed.
            Optional (with defaults): save_text=True, save_json=True,
            verbose=True, backend="sagemath".
            Missing required keys will raise when building the pipeline.
        instance_generator: Callable(seed) -> (problem, answer). Required.
        statistics_calculator: Optional callable(problem, answer) -> dict.
            Defaults to None (no per-sample statistics).

    Returns:
        DatasetPipeline instance configured with config and the given callables.
    """
    return cls(
        instance_generator=instance_generator,
        statistics_calculator=statistics_calculator,
        save_dir=config.save_dir,
        save_text=getattr(config, "save_text", True),
        save_json=getattr(config, "save_json", True),
        num_train_samples=config.num_train_samples,
        num_test_samples=config.num_test_samples,
        batch_size=config.batch_size,
        n_jobs=config.n_jobs,
        root_seed=config.root_seed,
        verbose=getattr(config, "verbose", True),
        backend=getattr(config, "backend", "sagemath"),
    )
```

### run

```
run() -> None
```

Run the pipeline: generate train/test data and write to save_dir.

Loads the backend (DatasetGenerator and DatasetWriter), then runs batch generation with the instance_generator and statistics_calculator, and writes outputs according to save_text and save_json.

Source code in `src/calt/dataset/pipeline.py`

```
def run(self) -> None:
    """Run the pipeline: generate train/test data and write to save_dir.

    Loads the backend (DatasetGenerator and DatasetWriter), then runs
    batch generation with the instance_generator and statistics_calculator,
    and writes outputs according to save_text and save_json.
    """
    DatasetGenerator, DatasetWriter = get_backend_classes(self.backend)

    # Initialize dataset generator
    dataset_generator = DatasetGenerator(
        backend="multiprocessing",
        n_jobs=self.n_jobs,
        verbose=self.verbose,
        root_seed=self.root_seed,
    )

    # Initialize writer
    dataset_writer = DatasetWriter(
        save_dir=self.save_dir,
        save_text=self.save_text,
        save_json=self.save_json,
    )

    # Generate datasets with batch processing
    dataset_generator.run(
        dataset_sizes={
            "train": self.num_train_samples,
            "test": self.num_test_samples,
        },
        batch_size=self.batch_size,
        instance_generator=self.instance_generator,
        statistics_calculator=self.statistics_calculator,
        dataset_writer=dataset_writer,
    )
```

Configuration for the dataset pipeline is done via the `dataset` block in `data.yaml`. For the option list, usage example, and YAML sample, see [Configuration](../configuration/).

## DatasetWriter

```
DatasetWriter(
    save_dir: str | None = None, save_text: bool = True, save_json: bool = True
)
```

Dataset writer for saving problem-answer pairs in multiple formats.

This class handles saving datasets with nested structure support up to 2 levels. It can save data in raw text (.txt) and JSON Lines (.jsonl) formats.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `INNER_SEP` | `str` | Separator for single-level lists (" | ") | | `OUTER_SEP` | `str` | Separator for nested lists (" || ") | | `save_dir` | `Path` | Base directory for saving datasets | | `save_text` | `bool` | Whether to save raw text files | | `save_json` | `bool` | Whether to save JSON Lines files | | `_file_handles` | `dict` | Dictionary to store open file handles |

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `save_dir` | `str | None` | Base directory for saving datasets. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Text files use "#" as separator between problem and answer, with nested structures joined by separators. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. JSON Lines files preserve the original nested structure format, with one sample per line. | `True` |

Usage

```
# Efficient batch processing with file handle management
writer = DatasetWriter(save_dir="./datasets")
writer.open("train")  # Open file handles once
try:
    for batch_idx, samples in enumerate(batches):
        writer.save_batch(samples, tag="train", batch_idx=batch_idx)
finally:
    writer.close("train")  # Close file handles

# Or use context manager
with DatasetWriter(save_dir="./datasets") as writer:
    writer.open("train")
    for batch_idx, samples in enumerate(batches):
        writer.save_batch(samples, tag="train", batch_idx=batch_idx)
    writer.close("train")

# Support for various dataset splits
writer.open("validation")  # Validation set
writer.open("dev")         # Development set
writer.open("eval")        # Evaluation set
```

Source code in `src/calt/dataset/utils/dataset_writer.py`

````
def __init__(
    self,
    save_dir: str | None = None,
    save_text: bool = True,
    save_json: bool = True,
) -> None:
    """
    Initialize dataset writer.

    Args:
        save_dir: Base directory for saving datasets. If None, uses current working directory.
        save_text: Whether to save raw text files. Text files use "#" as separator
                  between problem and answer, with nested structures joined by separators.
        save_json: Whether to save JSON Lines files. JSON Lines files preserve the original
                  nested structure format, with one sample per line.

    Usage:
        ```python
        # Efficient batch processing with file handle management
        writer = DatasetWriter(save_dir="./datasets")
        writer.open("train")  # Open file handles once
        try:
            for batch_idx, samples in enumerate(batches):
                writer.save_batch(samples, tag="train", batch_idx=batch_idx)
        finally:
            writer.close("train")  # Close file handles

        # Or use context manager
        with DatasetWriter(save_dir="./datasets") as writer:
            writer.open("train")
            for batch_idx, samples in enumerate(batches):
                writer.save_batch(samples, tag="train", batch_idx=batch_idx)
            writer.close("train")

        # Support for various dataset splits
        writer.open("validation")  # Validation set
        writer.open("dev")         # Development set
        writer.open("eval")        # Evaluation set
        ```
    """
    self.save_dir = Path(save_dir) if save_dir else Path.cwd()
    self.save_text = save_text
    self.save_json = save_json
    self.logger = logging.getLogger(__name__)
    self._file_handles: dict[
        str, dict[str, Any]
    ] = {}  # {tag: {file_type: file_handle}}
    TimedeltaDumper.add_representer(timedelta, timedelta_representer)
````
