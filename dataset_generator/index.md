# Dataset Generator

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and solutions, supports batch writing, and computes incremental statistics.

## Common (SageMath backend example)

### Generation flow

Base class for problem generators

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `backend` | `str` | Backend for parallel processing | `'multiprocessing'` | | `n_jobs` | `int` | Number of parallel jobs (-1 for all cores) | `-1` | | `verbose` | `bool` | Whether to display progress information | `True` | | `root_seed` | `int` | Root seed for reproducibility | `42` |

Source code in `src/calt/dataset_generator/sagemath/dataset_generator.py`

```
def __init__(
    self,
    backend: str = "multiprocessing",
    n_jobs: int = -1,
    verbose: bool = True,
    root_seed: int = 42,
):
    """
    Initialize problem generator.

    Args:
        backend: Backend for parallel processing
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Whether to display progress information
        root_seed: Root seed for reproducibility
    """

    self.backend = backend
    self.n_jobs = n_jobs
    self.verbose = verbose
    self.root_seed = root_seed

    # Configure logging only once at initialization
    self.logger = logger

    # Configure joblib logging to show progress but not overwhelm
    # Only set if not already configured
    joblib_logger = logging.getLogger("joblib")
    if joblib_logger.level == logging.NOTSET:
        joblib_logger.setLevel(logging.INFO)

    parallel_logger = logging.getLogger("joblib.Parallel")
    if parallel_logger.level == logging.NOTSET:
        parallel_logger.setLevel(logging.INFO)

```

## run

```
run(
    dataset_sizes: dict[str, int],
    problem_generator: Callable,
    statistics_calculator: Callable | None = None,
    dataset_writer: DatasetWriter | None = None,
    batch_size: int = 100000,
    save_dir: str | None = None,
    save_text: bool = True,
    save_json: bool = True,
)

```

Generate multiple datasets using parallel processing with batch writing.

This is the main entry point for dataset generation. It supports generating multiple datasets (train/test) simultaneously or separately, with efficient memory management through batch processing and parallel execution.

Key features:

- Parallel processing using joblib for high performance
- Batch-based memory management to handle large datasets
- Incremental statistics calculation to avoid memory issues
- Reproducible generation with unique seeds for each sample
- Support for nested data structures (up to 2 levels)
- Multiple output formats (pickle, text, JSON) via DatasetWriter

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `dataset_sizes` | `dict[str, int]` | Dictionary mapping dataset names to number of samples. Any string can be used as dataset name (e.g., "train", "test", "validation"). Duplicate names are not allowed. Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000} | *required* | | `problem_generator` | `Callable` | Function that generates (problem, solution) pair given a seed. Must accept a single integer seed parameter. | *required* | | `statistics_calculator` | `Callable | None` | Optional function to calculate sample-specific statistics. Must accept (problem, solution) and return dict or None. | `None` | | `dataset_writer` | `DatasetWriter | None` | DatasetWriter object for saving datasets to files. If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters. | `None` | | `batch_size` | `int` | Number of samples to process in each batch. Larger batches use more memory but may be more efficient for I/O operations. | `100000` | | `save_dir` | `str | None` | Base directory for saving datasets. Used only if dataset_writer is None. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Used only if dataset_writer is None. Text files use "#" as separator between problem and solution. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. Used only if dataset_writer is None. JSON Lines files preserve the original nested structure format. | `True` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If dataset_sizes is invalid or problem_generator is None | | `Exception` | If parallel processing fails |

Note

- Each sample gets a unique seed for reproducibility
- Progress is logged if verbose=True (set in **init**)
- Memory usage scales with batch_size, not total dataset size
- Statistics are calculated incrementally to handle large datasets
- If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

Examples:

```
>>> # Define problem generator function
>>> def polynomial_generator(seed):
...     import random
...     random.seed(seed)
...     # Generate random polynomial problem
...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
...     solution = sum(problem)
...     return problem, solution
>>>
>>> # Initialize dataset generator
>>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
>>>
>>> # Method 1: Automatic DatasetWriter creation
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     save_text=True,
...     save_json=True,
...     batch_size=100
... )
>>>
>>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
>>> from calt.dataset_generator.sagemath import DatasetWriter
>>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000},
...     problem_generator=polynomial_generator,
...     dataset_writer=writer,
...     batch_size=100
... )
>>>
>>> # Method 3: Generate datasets separately (if needed)
>>> generator.run(
...     dataset_sizes={"train": 10000},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     batch_size=100
... )
>>> generator.run(
...     dataset_sizes={"test": 1000, "validation": 500},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     batch_size=100
... )

```

Source code in `src/calt/dataset_generator/sagemath/dataset_generator.py`

```
def run(
    self,
    dataset_sizes: dict[str, int],
    problem_generator: Callable,
    statistics_calculator: Callable | None = None,
    dataset_writer: DatasetWriter | None = None,
    batch_size: int = 100000,
    save_dir: str | None = None,
    save_text: bool = True,
    save_json: bool = True,
):
    """
    Generate multiple datasets using parallel processing with batch writing.

    This is the main entry point for dataset generation. It supports generating
    multiple datasets (train/test) simultaneously or separately, with efficient
    memory management through batch processing and parallel execution.

    Key features:
    - Parallel processing using joblib for high performance
    - Batch-based memory management to handle large datasets
    - Incremental statistics calculation to avoid memory issues
    - Reproducible generation with unique seeds for each sample
    - Support for nested data structures (up to 2 levels)
    - Multiple output formats (pickle, text, JSON) via DatasetWriter

    Args:
        dataset_sizes: Dictionary mapping dataset names to number of samples.
                      Any string can be used as dataset name (e.g., "train", "test", "validation").
                      Duplicate names are not allowed.
                      Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000}
        problem_generator: Function that generates (problem, solution) pair given a seed.
                         Must accept a single integer seed parameter.
        statistics_calculator: Optional function to calculate sample-specific statistics.
                             Must accept (problem, solution) and return dict or None.
        dataset_writer: DatasetWriter object for saving datasets to files.
                      If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters.
        batch_size: Number of samples to process in each batch. Larger batches
                   use more memory but may be more efficient for I/O operations.
        save_dir: Base directory for saving datasets. Used only if dataset_writer is None.
                 If None, uses current working directory.
        save_text: Whether to save raw text files. Used only if dataset_writer is None.
                  Text files use "#" as separator between problem and solution.
        save_json: Whether to save JSON Lines files. Used only if dataset_writer is None.
                  JSON Lines files preserve the original nested structure format.

    Raises:
        ValueError: If dataset_sizes is invalid or problem_generator is None
        Exception: If parallel processing fails

    Note:
        - Each sample gets a unique seed for reproducibility
        - Progress is logged if verbose=True (set in __init__)
        - Memory usage scales with batch_size, not total dataset size
        - Statistics are calculated incrementally to handle large datasets
        - If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

    Examples:
        >>> # Define problem generator function
        >>> def polynomial_generator(seed):
        ...     import random
        ...     random.seed(seed)
        ...     # Generate random polynomial problem
        ...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
        ...     solution = sum(problem)
        ...     return problem, solution
        >>>
        >>> # Initialize dataset generator
        >>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
        >>>
        >>> # Method 1: Automatic DatasetWriter creation
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     save_text=True,
        ...     save_json=True,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
        >>> from calt.dataset_generator.sagemath import DatasetWriter
        >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000},
        ...     problem_generator=polynomial_generator,
        ...     dataset_writer=writer,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 3: Generate datasets separately (if needed)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     batch_size=100
        ... )
        >>> generator.run(
        ...     dataset_sizes={"test": 1000, "validation": 500},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     batch_size=100
        ... )
    """
    # Create DatasetWriter if not provided
    if dataset_writer is None:
        dataset_writer = DatasetWriter(
            save_dir=save_dir,
            save_text=save_text,
            save_json=save_json,
        )
        self.logger.info(f"save_dir: {dataset_writer.save_dir}")
        self.logger.info(f"Text output: {save_text}")
        self.logger.info(f"JSON output: {save_json}")

    # Prepare common arguments
    common_args = {
        "problem_generator": problem_generator,
        "statistics_calculator": statistics_calculator,
        "dataset_writer": dataset_writer,
        "batch_size": batch_size,
    }

    # Validate dataset_sizes
    if not isinstance(dataset_sizes, dict):
        raise ValueError("dataset_sizes must be a dictionary")

    if not dataset_sizes:
        raise ValueError("dataset_sizes cannot be empty")

    if problem_generator is None:
        raise ValueError("problem_generator must be provided")

    # Check for duplicate dataset names
    if len(dataset_sizes) != len(set(dataset_sizes.keys())):
        raise ValueError("Duplicate dataset names are not allowed")

    for dataset_name, num_samples in dataset_sizes.items():
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"Number of samples must be a positive integer, got {num_samples} for {dataset_name}"
            )

    # Log overall generation start
    self.logger.info(
        "=========================== Dataset generation ===========================\n"
    )
    self.logger.info(
        f"Starting dataset generation for {len(dataset_sizes)} dataset(s)"
    )
    self.logger.info(f"Dataset sizes: {dataset_sizes}\n")

    # Generate each dataset
    for dataset_name, num_samples in dataset_sizes.items():
        self._generate_dataset(
            tag=dataset_name, num_samples=num_samples, **common_args
        )

    self.logger.info("All datasets generated successfully!")
    self.logger.info(
        "==========================================================================\n"
    )

```

### Writing and statistics

Dataset writer for saving problem-solution pairs in multiple formats.

This class handles saving datasets with nested structure support up to 2 levels. It can save data in pickle (binary), raw text, and JSON Lines formats.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `INNER_SEP` | `str` | Separator for single-level lists (" | ") | | `OUTER_SEP` | `str` | Separator for nested lists (" || ") | | `save_dir` | `Path` | Base directory for saving datasets | | `save_text` | `bool` | Whether to save raw text files | | `save_json` | `bool` | Whether to save JSON Lines files | | `_file_handles` | `dict` | Dictionary to store open file handles |

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `save_dir` | `str | None` | Base directory for saving datasets. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Text files use "#" as separator between problem and solution, with nested structures joined by separators. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. JSON Lines files preserve the original nested structure format, with one sample per line. | `True` |

Note

Pickle files are always saved as they are the primary format for data loading. Text and JSON Lines files are optional and controlled by save_text and save_json flags.

Usage

## Efficient batch processing with file handle management

writer = DatasetWriter(save_dir="./datasets") writer.open("train") # Open file handles once try: for batch_idx, samples in enumerate(batches): writer.save_batch(samples, tag="train", batch_idx=batch_idx) finally: writer.close("train") # Close file handles

## Or use context manager

with DatasetWriter(save_dir="./datasets") as writer: writer.open("train") for batch_idx, samples in enumerate(batches): writer.save_batch(samples, tag="train", batch_idx=batch_idx) writer.close("train")

## Support for various dataset splits

writer.open("validation") # Validation set writer.open("dev") # Development set writer.open("eval") # Evaluation set

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
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
                  between problem and solution, with nested structures joined by separators.
        save_json: Whether to save JSON Lines files. JSON Lines files preserve the original
                  nested structure format, with one sample per line.

    Note:
        Pickle files are always saved as they are the primary format for data loading.
        Text and JSON Lines files are optional and controlled by save_text and save_json flags.

    Usage:
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
    """
    self.save_dir = Path(save_dir) if save_dir else Path.cwd()
    self.save_text = save_text
    self.save_json = save_json
    self.logger = logging.getLogger(__name__)
    self._file_handles: dict[
        str, dict[str, any]
    ] = {}  # {tag: {file_type: file_handle}}
    TimedeltaDumper.add_representer(timedelta, timedelta_representer)

```

## open

```
open(tag: str) -> None

```

Open file handles for the specified tag.

This method should be called before starting batch processing to avoid repeated file open/close operations.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def open(self, tag: str) -> None:
    """
    Open file handles for the specified tag.

    This method should be called before starting batch processing to avoid
    repeated file open/close operations.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid
    """
    self._validate_tag(tag)

    if tag in self._file_handles:
        self.logger.warning(f"File handles for tag '{tag}' are already open")
        return

    dataset_dir = self._create_dataset_dir()
    self._file_handles[tag] = {}

    # Create batch directory for pickle files
    batch_dir = dataset_dir / f"{tag}_batches"
    batch_dir.mkdir(exist_ok=True)
    self._file_handles[tag]["batch_dir"] = batch_dir
    self._file_handles[tag]["batch_count"] = 0

    # Open text file if enabled
    if self.save_text:
        raw_path = dataset_dir / f"{tag}_raw.txt"
        self._file_handles[tag]["text"] = open(raw_path, "w")

    # Open JSON Lines file if enabled
    if self.save_json:
        json_path = dataset_dir / f"{tag}_data.jsonl"
        self._file_handles[tag]["json"] = open(json_path, "w")

```

## close

```
close(tag: str) -> None

```

Close file handles for the specified tag.

This method should be called after finishing batch processing.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def close(self, tag: str) -> None:
    """
    Close file handles for the specified tag.

    This method should be called after finishing batch processing.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid
    """
    self._validate_tag(tag)

    if tag not in self._file_handles:
        self.logger.warning(f"No open file handles found for tag '{tag}'")
        return

    # Combine batch files into final pickle file
    if "batch_dir" in self._file_handles[tag]:
        self._combine_batch_files(tag)

    # Close all open file handles
    for file_type, file_handle in self._file_handles[tag].items():
        if hasattr(file_handle, "close"):  # Only close actual file handles
            try:
                file_handle.close()
            except Exception as e:
                self.logger.error(
                    f"Error closing {file_type} file for tag '{tag}': {e}"
                )

    del self._file_handles[tag]

```

## close_all

```
close_all() -> None

```

Close all open file handles.

This method should be called when the writer is no longer needed.

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def close_all(self) -> None:
    """
    Close all open file handles.

    This method should be called when the writer is no longer needed.
    """
    for tag in list(self._file_handles.keys()):
        self.close(tag)

```

## __enter__

```
__enter__()

```

Context manager entry.

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def __enter__(self):
    """Context manager entry."""
    return self

```

## __exit__

```
__exit__(exc_type, exc_val, exc_tb)

```

Context manager exit - close all files.

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - close all files."""
    self.close_all()

```

## save_batch

```
save_batch(
    samples: StringSampleList, tag: str = "train", batch_idx: int = 0
) -> None

```

Save a batch of samples to files in multiple formats.

This method saves samples in three formats:

1. Pickle (.pkl) - Binary format, always saved, used for loading
1. Raw text (.txt) - Human-readable format with separators (if save_text=True)
1. JSON Lines (.jsonl) - Structured format preserving nested structure (if save_json=True)

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `samples` | `StringSampleList` | List of (problem, solution) pairs in string format | *required* | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | `'train'` | | `batch_idx` | `int` | Batch index for incremental saving. Use 0 for first batch, subsequent batches will append to existing files. | `0` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid or samples contain invalid nested structures |

Examples:

```
>>> # Simple string samples (single problem-solution pairs)
>>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
>>> samples = [
...     ("x^2 + 2*x + 1", "(x + 1)^2"),
...     ("2*x^3 - 3*x^2", "x^2*(2*x - 3)"),
... ]
>>> # Creates: train_data.pkl, train_raw.txt, train_data.jsonl
>>> writer.save_batch(samples, tag="train", batch_idx=0)
>>>
>>> # 1 level nested structure samples (multiple problems/solutions)
>>> samples = [
...     (["x + y", "x - y"], ["2*x", "2*y"]),
...     (["x^2 + y^2", "x^2 - y^2"], ["2*x^2", "2*y^2"]),
... ]
>>> # Text output: "x + y | x - y # 2*x | 2*y"
>>> writer.save_batch(samples, tag="test", batch_idx=0)
>>>
>>> # 2 level nested structure samples (complex nested problems)
>>> samples = [
...     ([["x", "y"], ["z", "w"]], [["x", "z"], ["y", "w"]]),
...     ([["x + y", "x - y"], ["z + w", "z - w"]], [["x + y", "z + w"], ["x - y", "z - w"]]),
... ]
>>> # Text output: "x | y || z | w # x | z || y | w"
>>> writer.save_batch(samples, tag="test", batch_idx=0)
>>>
>>> # Append more samples to existing dataset
>>> more_samples = [
...     ([["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]),
...     ([["e", "f"], ["g", "h"]], [["e", "g"], ["f", "h"]]),
... ]
>>> # Appends to existing files instead of overwriting
>>> writer.save_batch(more_samples, tag="train", batch_idx=1)

```

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def save_batch(
    self,
    samples: StringSampleList,
    tag: str = "train",
    batch_idx: int = 0,
) -> None:
    """
    Save a batch of samples to files in multiple formats.

    This method saves samples in three formats:
    1. Pickle (.pkl) - Binary format, always saved, used for loading
    2. Raw text (.txt) - Human-readable format with separators (if save_text=True)
    3. JSON Lines (.jsonl) - Structured format preserving nested structure (if save_json=True)

    Args:
        samples: List of (problem, solution) pairs in string format
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")
        batch_idx: Batch index for incremental saving. Use 0 for first batch,
                  subsequent batches will append to existing files.

    Raises:
        ValueError: If tag is invalid or samples contain invalid nested structures

    Examples:
        >>> # Simple string samples (single problem-solution pairs)
        >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
        >>> samples = [
        ...     ("x^2 + 2*x + 1", "(x + 1)^2"),
        ...     ("2*x^3 - 3*x^2", "x^2*(2*x - 3)"),
        ... ]
        >>> # Creates: train_data.pkl, train_raw.txt, train_data.jsonl
        >>> writer.save_batch(samples, tag="train", batch_idx=0)
        >>>
        >>> # 1 level nested structure samples (multiple problems/solutions)
        >>> samples = [
        ...     (["x + y", "x - y"], ["2*x", "2*y"]),
        ...     (["x^2 + y^2", "x^2 - y^2"], ["2*x^2", "2*y^2"]),
        ... ]
        >>> # Text output: "x + y | x - y # 2*x | 2*y"
        >>> writer.save_batch(samples, tag="test", batch_idx=0)
        >>>
        >>> # 2 level nested structure samples (complex nested problems)
        >>> samples = [
        ...     ([["x", "y"], ["z", "w"]], [["x", "z"], ["y", "w"]]),
        ...     ([["x + y", "x - y"], ["z + w", "z - w"]], [["x + y", "z + w"], ["x - y", "z - w"]]),
        ... ]
        >>> # Text output: "x | y || z | w # x | z || y | w"
        >>> writer.save_batch(samples, tag="test", batch_idx=0)
        >>>
        >>> # Append more samples to existing dataset
        >>> more_samples = [
        ...     ([["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]),
        ...     ([["e", "f"], ["g", "h"]], [["e", "g"], ["f", "h"]]),
        ... ]
        >>> # Appends to existing files instead of overwriting
        >>> writer.save_batch(more_samples, tag="train", batch_idx=1)
    """
    self._validate_tag(tag)

    # Validate samples
    if not samples:
        self.logger.warning(
            "Empty samples list provided - no files will be created"
        )
        return

    # Check if file handles are open for this tag
    if tag not in self._file_handles:
        # Fallback to old method if file handles are not open
        self._save_batch_legacy(samples, tag, batch_idx)
        return

    # Save binary data (pickle format) - save batch individually
    batch_file = (
        self._file_handles[tag]["batch_dir"]
        / f"batch_{self._file_handles[tag]['batch_count']:06d}.pkl"
    )
    with open(batch_file, "wb") as f:
        pickle.dump(samples, f)
    self._file_handles[tag]["batch_count"] += 1

    # Save raw text data (optional)
    if self.save_text:
        for problem_str, solution_str in samples:
            formatted_line = self._format_sample_strings(problem_str, solution_str)
            self._file_handles[tag]["text"].write(f"{formatted_line}\n")
        self._file_handles[tag]["text"].flush()

    # Save JSON Lines data (optional)
    if self.save_json:
        for problem_str, solution_str in samples:
            json_data = self._get_json_data(problem_str, solution_str)
            json_line = json.dumps(json_data, ensure_ascii=False)
            self._file_handles[tag]["json"].write(f"{json_line}\n")
        self._file_handles[tag]["json"].flush()

```

## save_final_statistics

```
save_final_statistics(statistics: StatisticsDict, tag: str = 'train') -> None

```

Save final overall statistics to YAML file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `statistics` | `StatisticsDict` | Dictionary containing dataset statistics | *required* | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | `'train'` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Note

Statistics are saved in YAML format for human readability. The file is named "{tag}\_stats.yaml" in the dataset directory.

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def save_final_statistics(
    self,
    statistics: StatisticsDict,
    tag: str = "train",
) -> None:
    """
    Save final overall statistics to YAML file.

    Args:
        statistics: Dictionary containing dataset statistics
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid

    Note:
        Statistics are saved in YAML format for human readability.
        The file is named "{tag}_stats.yaml" in the dataset directory.
    """
    self._validate_tag(tag)
    dataset_dir = self._create_dataset_dir()

    stats_path = dataset_dir / f"{tag}_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(
            statistics,
            f,
            Dumper=TimedeltaDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=4,
        )

```

## load_dataset

```
load_dataset(tag: str) -> StringSampleList

```

Load dataset from pickle file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Returns:

| Type | Description | | --- | --- | | `StringSampleList` | List of (problem, solution) pairs in string format |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid | | `FileNotFoundError` | If the pickle file doesn't exist |

Examples:

```
>>> samples = writer.load_dataset("train")
>>> print(f"Loaded {len(samples)} samples")
>>> for problem, solution in samples[:3]:
...     print(f"Problem: {problem}, Solution: {solution}")

```

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def load_dataset(self, tag: str) -> StringSampleList:
    """
    Load dataset from pickle file.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Returns:
        List of (problem, solution) pairs in string format

    Raises:
        ValueError: If tag is invalid
        FileNotFoundError: If the pickle file doesn't exist

    Examples:
        >>> samples = writer.load_dataset("train")
        >>> print(f"Loaded {len(samples)} samples")
        >>> for problem, solution in samples[:3]:
        ...     print(f"Problem: {problem}, Solution: {solution}")
    """
    self._validate_tag(tag)
    pickle_path = self.save_dir / f"{tag}_data.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)

```

## load_dataset_jsonl

```
load_dataset_jsonl(tag: str) -> StringSampleList

```

Load dataset from JSON Lines file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Returns:

| Type | Description | | --- | --- | | `StringSampleList` | List of (problem, solution) pairs in string format |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid | | `FileNotFoundError` | If the JSON Lines file doesn't exist |

Examples:

```
>>> samples = writer.load_dataset_jsonl("train")
>>> print(f"Loaded {len(samples)} samples")
>>> for problem, solution in samples[:3]:
...     print(f"Problem: {problem}, Solution: {solution}")

```

Source code in `src/calt/dataset_generator/sagemath/utils/dataset_writer.py`

```
def load_dataset_jsonl(self, tag: str) -> StringSampleList:
    """
    Load dataset from JSON Lines file.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Returns:
        List of (problem, solution) pairs in string format

    Raises:
        ValueError: If tag is invalid
        FileNotFoundError: If the JSON Lines file doesn't exist

    Examples:
        >>> samples = writer.load_dataset_jsonl("train")
        >>> print(f"Loaded {len(samples)} samples")
        >>> for problem, solution in samples[:3]:
        ...     print(f"Problem: {problem}, Solution: {solution}")
    """
    self._validate_tag(tag)
    jsonl_path = self.save_dir / f"{tag}_data.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSON Lines file not found: {jsonl_path}")

    samples = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                problem = data["problem"]
                solution = data["solution"]
                samples.append((problem, solution))
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    return samples

```

Memory-efficient statistics calculator that uses incremental computation.

This calculator avoids storing all data in memory by computing statistics incrementally as batches are processed using Welford's online algorithm for numerical stability and memory efficiency. All standard deviations are calculated as population standard deviations.

Source code in `src/calt/dataset_generator/sagemath/utils/statistics_calculator.py`

```
def __init__(self):
    """Initialize incremental sample statistics calculator."""
    self.runtime_stats = IncrementalStatistics()
    self.sample_stats = {}  # Store aggregated sample statistics by category

```

## update_batch

```
update_batch(
    runtimes: list[float],
    batch_sample_stats: list[dict[str, dict[str, int | float]]],
) -> None

```

Update statistics with a batch of results using Welford's online algorithm.

This method processes each sample individually, updating both runtime statistics and sample-specific statistics incrementally for better control and efficiency.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `runtimes` | `list[float]` | List of runtime values for each sample in the batch | *required* | | `batch_sample_stats` | `list[dict[str, dict[str, int | float]]]` | List of sample statistics dictionaries for the current batch. Each dictionary has the structure: {"category1": {"metric1": value1, ...}, "category2": {"metric1": value1, ...}} Example: [{"problem": {"total_degree": 2, "num_polynomials": 3}, "solution": {"total_degree": 3, "num_polynomials": 3}}, {"problem": {"total_degree": 5, "num_polynomials": 4}, "solution": {"total_degree": 8, "num_polynomials": 4}}, ...] | *required* |

Source code in `src/calt/dataset_generator/sagemath/utils/statistics_calculator.py`

```
def update_batch(
    self,
    runtimes: list[float],
    batch_sample_stats: list[dict[str, dict[str, int | float]]],
) -> None:
    """
    Update statistics with a batch of results using Welford's online algorithm.

    This method processes each sample individually, updating both runtime
    statistics and sample-specific statistics incrementally for better
    control and efficiency.

    Args:
        runtimes: List of runtime values for each sample in the batch
        batch_sample_stats: List of sample statistics dictionaries for the current batch.
                           Each dictionary has the structure:
                           {"category1": {"metric1": value1, ...},
                            "category2": {"metric1": value1, ...}}
                           Example:
                           [{"problem": {"total_degree": 2, "num_polynomials": 3},
                             "solution": {"total_degree": 3, "num_polynomials": 3}},
                            {"problem": {"total_degree": 5, "num_polynomials": 4},
                             "solution": {"total_degree": 8, "num_polynomials": 4}},
                            ...]
    """
    # Update runtime statistics
    for runtime in runtimes:
        self.runtime_stats.update(runtime)

    # Update sample statistics
    for stats in batch_sample_stats:
        # Update each numeric sample statistic incrementally
        for category, category_stats in stats.items():
            if isinstance(category_stats, dict):
                # Handle nested structure like {"problem": {...}, "solution": {...}}
                if category not in self.sample_stats:
                    self.sample_stats[category] = {}

                for stat_name, value in category_stats.items():
                    if isinstance(value, (int, float)):
                        if stat_name not in self.sample_stats[category]:
                            self.sample_stats[category][stat_name] = (
                                IncrementalStatistics()
                            )
                        self.sample_stats[category][stat_name].update(float(value))

            elif isinstance(category_stats, (int, float)):
                # Handle flat structure
                if category not in self.sample_stats:
                    self.sample_stats[category] = IncrementalStatistics()
                self.sample_stats[category].update(float(category_stats))

```

## get_overall_statistics

```
get_overall_statistics(total_time: float, num_samples: int) -> dict[str, Any]

```

Get overall statistics.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `total_time` | `float` | Total processing time | *required* | | `num_samples` | `int` | Total number of samples | *required* |

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | Dictionary containing overall statistics with the structure: | | `dict[str, Any]` | { "total_time": float, "num_samples": int, "samples_per_second": float, "generation_time": {"mean": float, "std": float, "min": float, "max": float}, "problem_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...}, "solution_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...} | | `dict[str, Any]` | } |

Source code in `src/calt/dataset_generator/sagemath/utils/statistics_calculator.py`

```
def get_overall_statistics(
    self, total_time: float, num_samples: int
) -> dict[str, Any]:
    """
    Get overall statistics.

    Args:
        total_time: Total processing time
        num_samples: Total number of samples

    Returns:
        Dictionary containing overall statistics with the structure:
        {
            "total_time": float,
            "num_samples": int,
            "samples_per_second": float,
            "generation_time": {"mean": float, "std": float, "min": float, "max": float},
            "problem_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...},
            "solution_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...}
        }
    """
    runtime_stats = self.runtime_stats.get_statistics()

    overall_stats = {
        "total_time": total_time,
        "num_samples": num_samples,
        "samples_per_second": num_samples / total_time if total_time > 0 else 0.0,
        "generation_time": {
            "mean": runtime_stats["mean"],
            "std": runtime_stats["std"],
            "min": runtime_stats["min"],
            "max": runtime_stats["max"],
        },
    }

    # Add sample statistics by category
    for category, category_stats in self.sample_stats.items():
        if isinstance(category_stats, dict):
            # Handle nested structure like {"problem": {...}, "solution": {...}}
            overall_stats[f"{category}_stats"] = {
                stat_name: stat_calc.get_statistics()
                for stat_name, stat_calc in category_stats.items()
            }
        else:
            # Handle flat structure
            overall_stats[f"{category}_stats"] = category_stats.get_statistics()

    return overall_stats

```

### Sampling

Generator for random polynomials with specific constraints

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `symbols` | `str | None` | Symbols of polynomial ring (required if ring is None) | `None` | | `field_str` | `str | None` | Field of polynomial ring (required if ring is None) | `None` | | `order` | `str | TermOrder | None` | Order of polynomial ring (required if ring is None) | `None` | | `ring` | `Any` | PolynomialRing object (alternative to symbols/field_str/order) | `None` | | `max_num_terms` | `int | None` | Maximum number of terms in polynomial. If None, all possible terms are allowed. | `10` | | `max_degree` | `int` | Maximum degree of polynomial | `5` | | `min_degree` | `int` | Minimum degree of polynomial | `0` | | `max_coeff` | `int | None` | Maximum coefficient value (used for RR and ZZ) | `None` | | `num_bound` | `int | None` | Maximum absolute value of coefficients (used for QQ) | `None` | | `degree_sampling` | `str` | How to sample degree ('uniform' or 'fixed') | `'uniform'` | | `term_sampling` | `str` | How to sample number of terms ('uniform' or 'fixed') | `'uniform'` | | `strictly_conditioned` | `bool` | Whether to strictly enforce conditions | `True` | | `nonzero_instance` | `bool` | Whether to enforce non-zero instance | `True` | | `nonzero_coeff` | `bool` | Whether to exclude zero coefficients during coefficient generation | `False` | | `max_attempts` | `int` | Maximum number of attempts to generate a polynomial satisfying conditions | `1000` |

Source code in `src/calt/dataset_generator/sagemath/utils/polynomial_sampler.py`

```
def __init__(
    self,
    symbols: str | None = None,
    field_str: str | None = None,
    order: str | TermOrder | None = None,
    ring: Any = None,
    max_num_terms: int | None = 10,
    max_degree: int = 5,
    min_degree: int = 0,
    degree_sampling: str = "uniform",  # 'uniform' or 'fixed'
    term_sampling: str = "uniform",  # 'uniform' or 'fixed'
    max_coeff: int | None = None,  # Used for RR and ZZ
    num_bound: int | None = None,  # Used for QQ
    strictly_conditioned: bool = True,
    nonzero_instance: bool = True,
    nonzero_coeff: bool = False,  # Whether to exclude zero coefficients
    max_attempts: int = 1000,
):
    """
    Initialize polynomial sampler

    Args:
        symbols: Symbols of polynomial ring (required if ring is None)
        field_str: Field of polynomial ring (required if ring is None)
        order: Order of polynomial ring (required if ring is None)
        ring: PolynomialRing object (alternative to symbols/field_str/order)
        max_num_terms: Maximum number of terms in polynomial. If None, all possible terms are allowed.
        max_degree: Maximum degree of polynomial
        min_degree: Minimum degree of polynomial
        max_coeff: Maximum coefficient value (used for RR and ZZ)
        num_bound: Maximum absolute value of coefficients (used for QQ)
        degree_sampling: How to sample degree ('uniform' or 'fixed')
        term_sampling: How to sample number of terms ('uniform' or 'fixed')
        strictly_conditioned: Whether to strictly enforce conditions
        nonzero_instance: Whether to enforce non-zero instance
        nonzero_coeff: Whether to exclude zero coefficients during coefficient generation
        max_attempts: Maximum number of attempts to generate a polynomial satisfying conditions
    """
    # Validate input parameters
    if ring is not None:
        if symbols is not None or field_str is not None or order is not None:
            raise ValueError("Cannot specify both ring and symbols/field_str/order")
        self.ring = ring
        self.symbols = None
        self.field_str = None
        self.order = None
    else:
        if symbols is None or field_str is None or order is None:
            raise ValueError(
                "Must specify either ring or all of symbols/field_str/order"
            )
        self.ring = None
        self.symbols = symbols
        self.field_str = field_str
        self.order = order

    self.max_num_terms = max_num_terms
    self.max_degree = max_degree
    self.min_degree = min_degree
    self.max_coeff = max_coeff
    self.num_bound = num_bound
    self.degree_sampling = degree_sampling
    self.term_sampling = term_sampling
    self.strictly_conditioned = strictly_conditioned
    self.nonzero_instance = nonzero_instance
    self.nonzero_coeff = nonzero_coeff
    self.max_attempts = max_attempts

```

## get_field

```
get_field()

```

Convert field_str to actual sympy domain object

Source code in `src/calt/dataset_generator/sagemath/utils/polynomial_sampler.py`

```
def get_field(self):
    """Convert field_str to actual sympy domain object"""
    if self.ring is not None:
        return self.ring.base_ring()

    # Standard field mapping
    standard_fields = {"QQ": QQ, "RR": RR, "ZZ": ZZ}
    if self.field_str in standard_fields:
        return standard_fields[self.field_str]

    # Finite field handling
    if not self.field_str.startswith("GF"):
        raise ValueError(f"Unsupported field: {self.field_str}")

    try:
        # Extract field size based on format
        p = int(
            self.field_str[3:-1]
            if self.field_str.startswith("GF(")
            else self.field_str[2:]
        )

        if p <= 1:
            raise ValueError(f"Field size must be greater than 1: {p}")
        return GF(p)
    except ValueError as e:
        raise ValueError(f"Unsupported field: {self.field_str}") from e

```

## get_ring

```
get_ring() -> PolynomialRing

```

Generate polynomial ring

Returns:

| Name | Type | Description | | --- | --- | --- | | `PolynomialRing` | `PolynomialRing` | Generated polynomial ring |

Source code in `src/calt/dataset_generator/sagemath/utils/polynomial_sampler.py`

```
def get_ring(self) -> PolynomialRing:
    """
    Generate polynomial ring

    Returns:
        PolynomialRing: Generated polynomial ring
    """
    if self.ring is not None:
        return self.ring

    R = PolynomialRing(self.get_field(), self.symbols, order=self.order)
    return R

```

## sample

```
sample(
    num_samples: int = 1,
    size: tuple[int, int] | None = None,
    density: float = 1.0,
    matrix_type: str | None = None,
) -> list[MPolynomial_libsingular] | list[matrix]

```

Generate random polynomial samples

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `num_samples` | `int` | Number of samples to generate | `1` | | `size` | `tuple[int, int] | None` | If provided, generate matrix of polynomials with given size | `None` | | `density` | `float` | Probability of non-zero entries in matrix | `1.0` | | `matrix_type` | `str | None` | Special matrix type (e.g., 'unimodular_upper_triangular') | `None` |

Returns:

| Type | Description | | --- | --- | | `list[MPolynomial_libsingular] | list[matrix]` | List of polynomials or polynomial matrices |

Source code in `src/calt/dataset_generator/sagemath/utils/polynomial_sampler.py`

```
def sample(
    self,
    num_samples: int = 1,
    size: tuple[int, int] | None = None,
    density: float = 1.0,
    matrix_type: str | None = None,
) -> list[MPolynomial_libsingular] | list[matrix]:
    """
    Generate random polynomial samples

    Args:
        num_samples: Number of samples to generate
        size: If provided, generate matrix of polynomials with given size
        density: Probability of non-zero entries in matrix
        matrix_type: Special matrix type (e.g., 'unimodular_upper_triangular')

    Returns:
        List of polynomials or polynomial matrices
    """
    if size is not None:
        return [
            self._sample_matrix(size, density, matrix_type)
            for _ in range(num_samples)
        ]
    else:
        return [self._sample_polynomial() for _ in range(num_samples)]

```

## Common (SymPy backend example)

### Generation flow

Base class for problem generators

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `backend` | `str` | Backend for parallel processing | `'multiprocessing'` | | `n_jobs` | `int` | Number of parallel jobs (-1 for all cores) | `-1` | | `verbose` | `bool` | Whether to display progress information | `True` | | `root_seed` | `int` | Root seed for reproducibility | `42` |

Source code in `src/calt/dataset_generator/sympy/dataset_generator.py`

```
def __init__(
    self,
    backend: str = "multiprocessing",
    n_jobs: int = -1,
    verbose: bool = True,
    root_seed: int = 42,
):
    """
    Initialize problem generator.

    Args:
        backend: Backend for parallel processing
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Whether to display progress information
        root_seed: Root seed for reproducibility
    """

    self.backend = backend
    self.n_jobs = n_jobs
    self.verbose = verbose
    self.root_seed = root_seed

    # Configure logging only once at initialization
    self.logger = logger

    # Configure joblib logging to show progress but not overwhelm
    # Only set if not already configured
    joblib_logger = logging.getLogger("joblib")
    if joblib_logger.level == logging.NOTSET:
        joblib_logger.setLevel(logging.INFO)

    parallel_logger = logging.getLogger("joblib.Parallel")
    if parallel_logger.level == logging.NOTSET:
        parallel_logger.setLevel(logging.INFO)

```

## run

```
run(
    dataset_sizes: dict[str, int],
    problem_generator: Callable,
    statistics_calculator: Callable | None = None,
    dataset_writer: DatasetWriter | None = None,
    batch_size: int = 100000,
    save_dir: str | None = None,
    save_text: bool = True,
    save_json: bool = True,
)

```

Generate multiple datasets using parallel processing with batch writing.

This is the main entry point for dataset generation. It supports generating multiple datasets (train/test) simultaneously or separately, with efficient memory management through batch processing and parallel execution.

Key features:

- Parallel processing using joblib for high performance
- Batch-based memory management to handle large datasets
- Incremental statistics calculation to avoid memory issues
- Reproducible generation with unique seeds for each sample
- Support for nested data structures (up to 2 levels)
- Multiple output formats (pickle, text, JSON) via DatasetWriter

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `dataset_sizes` | `dict[str, int]` | Dictionary mapping dataset names to number of samples. Any string can be used as dataset name (e.g., "train", "test", "validation"). Duplicate names are not allowed. Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000} | *required* | | `problem_generator` | `Callable` | Function that generates (problem, solution) pair given a seed. Must accept a single integer seed parameter. | *required* | | `statistics_calculator` | `Callable | None` | Optional function to calculate sample-specific statistics. Must accept (problem, solution) and return dict or None. | `None` | | `dataset_writer` | `DatasetWriter | None` | DatasetWriter object for saving datasets to files. If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters. | `None` | | `batch_size` | `int` | Number of samples to process in each batch. Larger batches use more memory but may be more efficient for I/O operations. | `100000` | | `save_dir` | `str | None` | Base directory for saving datasets. Used only if dataset_writer is None. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Used only if dataset_writer is None. Text files use "#" as separator between problem and solution. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. Used only if dataset_writer is None. JSON Lines files preserve the original nested structure format. | `True` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If dataset_sizes is invalid or problem_generator is None | | `Exception` | If parallel processing fails |

Note

- Each sample gets a unique seed for reproducibility
- Progress is logged if verbose=True (set in **init**)
- Memory usage scales with batch_size, not total dataset size
- Statistics are calculated incrementally to handle large datasets
- If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

Examples:

```
>>> # Define problem generator function
>>> def polynomial_generator(seed):
...     import random
...     random.seed(seed)
...     # Generate random polynomial problem
...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
...     solution = sum(problem)
...     return problem, solution
>>>
>>> # Initialize dataset generator
>>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
>>>
>>> # Method 1: Automatic DatasetWriter creation
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     save_text=True,
...     save_json=True,
...     batch_size=100
... )
>>>
>>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
>>> from calt.dataset_generator.sympy import DatasetWriter
>>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000},
...     problem_generator=polynomial_generator,
...     dataset_writer=writer,
...     batch_size=100
... )
>>>
>>> # Method 3: Generate datasets separately (if needed)
>>> generator.run(
...     dataset_sizes={"train": 10000},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     batch_size=100
... )
>>> generator.run(
...     dataset_sizes={"test": 1000, "validation": 500},
...     problem_generator=polynomial_generator,
...     save_dir="./datasets",
...     batch_size=100
... )

```

Source code in `src/calt/dataset_generator/sympy/dataset_generator.py`

```
def run(
    self,
    dataset_sizes: dict[str, int],
    problem_generator: Callable,
    statistics_calculator: Callable | None = None,
    dataset_writer: DatasetWriter | None = None,
    batch_size: int = 100000,
    save_dir: str | None = None,
    save_text: bool = True,
    save_json: bool = True,
):
    """
    Generate multiple datasets using parallel processing with batch writing.

    This is the main entry point for dataset generation. It supports generating
    multiple datasets (train/test) simultaneously or separately, with efficient
    memory management through batch processing and parallel execution.

    Key features:
    - Parallel processing using joblib for high performance
    - Batch-based memory management to handle large datasets
    - Incremental statistics calculation to avoid memory issues
    - Reproducible generation with unique seeds for each sample
    - Support for nested data structures (up to 2 levels)
    - Multiple output formats (pickle, text, JSON) via DatasetWriter

    Args:
        dataset_sizes: Dictionary mapping dataset names to number of samples.
                      Any string can be used as dataset name (e.g., "train", "test", "validation").
                      Duplicate names are not allowed.
                      Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000}
        problem_generator: Function that generates (problem, solution) pair given a seed.
                         Must accept a single integer seed parameter.
        statistics_calculator: Optional function to calculate sample-specific statistics.
                             Must accept (problem, solution) and return dict or None.
        dataset_writer: DatasetWriter object for saving datasets to files.
                      If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters.
        batch_size: Number of samples to process in each batch. Larger batches
                   use more memory but may be more efficient for I/O operations.
        save_dir: Base directory for saving datasets. Used only if dataset_writer is None.
                 If None, uses current working directory.
        save_text: Whether to save raw text files. Used only if dataset_writer is None.
                  Text files use "#" as separator between problem and solution.
        save_json: Whether to save JSON Lines files. Used only if dataset_writer is None.
                  JSON Lines files preserve the original nested structure format.

    Raises:
        ValueError: If dataset_sizes is invalid or problem_generator is None
        Exception: If parallel processing fails

    Note:
        - Each sample gets a unique seed for reproducibility
        - Progress is logged if verbose=True (set in __init__)
        - Memory usage scales with batch_size, not total dataset size
        - Statistics are calculated incrementally to handle large datasets
        - If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

    Examples:
        >>> # Define problem generator function
        >>> def polynomial_generator(seed):
        ...     import random
        ...     random.seed(seed)
        ...     # Generate random polynomial problem
        ...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
        ...     solution = sum(problem)
        ...     return problem, solution
        >>>
        >>> # Initialize dataset generator
        >>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
        >>>
        >>> # Method 1: Automatic DatasetWriter creation
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     save_text=True,
        ...     save_json=True,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
        >>> from calt.dataset_generator.sympy import DatasetWriter
        >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000},
        ...     problem_generator=polynomial_generator,
        ...     dataset_writer=writer,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 3: Generate datasets separately (if needed)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     batch_size=100
        ... )
        >>> generator.run(
        ...     dataset_sizes={"test": 1000, "validation": 500},
        ...     problem_generator=polynomial_generator,
        ...     save_dir="./datasets",
        ...     batch_size=100
        ... )
    """
    # Create DatasetWriter if not provided
    if dataset_writer is None:
        dataset_writer = DatasetWriter(
            save_dir=save_dir,
            save_text=save_text,
            save_json=save_json,
        )
        self.logger.info(f"save_dir: {dataset_writer.save_dir}")
        self.logger.info(f"Text output: {save_text}")
        self.logger.info(f"JSON output: {save_json}")

    # Prepare common arguments
    common_args = {
        "problem_generator": problem_generator,
        "statistics_calculator": statistics_calculator,
        "dataset_writer": dataset_writer,
        "batch_size": batch_size,
    }

    # Validate dataset_sizes
    if not isinstance(dataset_sizes, dict):
        raise ValueError("dataset_sizes must be a dictionary")

    if not dataset_sizes:
        raise ValueError("dataset_sizes cannot be empty")

    if problem_generator is None:
        raise ValueError("problem_generator must be provided")

    # Check for duplicate dataset names
    if len(dataset_sizes) != len(set(dataset_sizes.keys())):
        raise ValueError("Duplicate dataset names are not allowed")

    for dataset_name, num_samples in dataset_sizes.items():
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"Number of samples must be a positive integer, got {num_samples} for {dataset_name}"
            )

    # Log overall generation start
    self.logger.info(
        "=========================== Dataset generation ===========================\n"
    )
    self.logger.info(
        f"Starting dataset generation for {len(dataset_sizes)} dataset(s)"
    )
    self.logger.info(f"Dataset sizes: {dataset_sizes}\n")

    # Generate each dataset
    for dataset_name, num_samples in dataset_sizes.items():
        self._generate_dataset(
            tag=dataset_name, num_samples=num_samples, **common_args
        )

    self.logger.info("All datasets generated successfully!")
    self.logger.info(
        "==========================================================================\n"
    )

```

### Writing and statistics

Dataset writer for saving problem-solution pairs in multiple formats.

This class handles saving datasets with nested structure support up to 2 levels. It can save data in pickle (binary), raw text, and JSON Lines formats.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `INNER_SEP` | `str` | Separator for single-level lists (" | ") | | `OUTER_SEP` | `str` | Separator for nested lists (" || ") | | `save_dir` | `Path` | Base directory for saving datasets | | `save_text` | `bool` | Whether to save raw text files | | `save_json` | `bool` | Whether to save JSON Lines files | | `_file_handles` | `dict` | Dictionary to store open file handles |

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `save_dir` | `str | None` | Base directory for saving datasets. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Text files use "#" as separator between problem and solution, with nested structures joined by separators. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. JSON Lines files preserve the original nested structure format, with one sample per line. | `True` |

Note

Pickle files are always saved as they are the primary format for data loading. Text and JSON Lines files are optional and controlled by save_text and save_json flags.

Usage

## Efficient batch processing with file handle management

writer = DatasetWriter(save_dir="./datasets") writer.open("train") # Open file handles once try: for batch_idx, samples in enumerate(batches): writer.save_batch(samples, tag="train", batch_idx=batch_idx) finally: writer.close("train") # Close file handles

## Or use context manager

with DatasetWriter(save_dir="./datasets") as writer: writer.open("train") for batch_idx, samples in enumerate(batches): writer.save_batch(samples, tag="train", batch_idx=batch_idx) writer.close("train")

## Support for various dataset splits

writer.open("validation") # Validation set writer.open("dev") # Development set writer.open("eval") # Evaluation set

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
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
                  between problem and solution, with nested structures joined by separators.
        save_json: Whether to save JSON Lines files. JSON Lines files preserve the original
                  nested structure format, with one sample per line.

    Note:
        Pickle files are always saved as they are the primary format for data loading.
        Text and JSON Lines files are optional and controlled by save_text and save_json flags.

    Usage:
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
    """
    self.save_dir = Path(save_dir) if save_dir else Path.cwd()
    self.save_text = save_text
    self.save_json = save_json
    self.logger = logging.getLogger(__name__)
    self._file_handles: dict[
        str, dict[str, any]
    ] = {}  # {tag: {file_type: file_handle}}
    TimedeltaDumper.add_representer(timedelta, timedelta_representer)

```

## open

```
open(tag: str) -> None

```

Open file handles for the specified tag.

This method should be called before starting batch processing to avoid repeated file open/close operations.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def open(self, tag: str) -> None:
    """
    Open file handles for the specified tag.

    This method should be called before starting batch processing to avoid
    repeated file open/close operations.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid
    """
    self._validate_tag(tag)

    if tag in self._file_handles:
        self.logger.warning(f"File handles for tag '{tag}' are already open")
        return

    dataset_dir = self._create_dataset_dir()
    self._file_handles[tag] = {}

    # Create batch directory for pickle files
    batch_dir = dataset_dir / f"{tag}_batches"
    batch_dir.mkdir(exist_ok=True)
    self._file_handles[tag]["batch_dir"] = batch_dir
    self._file_handles[tag]["batch_count"] = 0

    # Open text file if enabled
    if self.save_text:
        raw_path = dataset_dir / f"{tag}_raw.txt"
        self._file_handles[tag]["text"] = open(raw_path, "w")

    # Open JSON Lines file if enabled
    if self.save_json:
        json_path = dataset_dir / f"{tag}_data.jsonl"
        self._file_handles[tag]["json"] = open(json_path, "w")

```

## close

```
close(tag: str) -> None

```

Close file handles for the specified tag.

This method should be called after finishing batch processing.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def close(self, tag: str) -> None:
    """
    Close file handles for the specified tag.

    This method should be called after finishing batch processing.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid
    """
    self._validate_tag(tag)

    if tag not in self._file_handles:
        self.logger.warning(f"No open file handles found for tag '{tag}'")
        return

    # Combine batch files into final pickle file
    if "batch_dir" in self._file_handles[tag]:
        self._combine_batch_files(tag)

    # Close all open file handles
    for file_type, file_handle in self._file_handles[tag].items():
        if hasattr(file_handle, "close"):  # Only close actual file handles
            try:
                file_handle.close()
            except Exception as e:
                self.logger.error(
                    f"Error closing {file_type} file for tag '{tag}': {e}"
                )

    del self._file_handles[tag]

```

## close_all

```
close_all() -> None

```

Close all open file handles.

This method should be called when the writer is no longer needed.

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def close_all(self) -> None:
    """
    Close all open file handles.

    This method should be called when the writer is no longer needed.
    """
    for tag in list(self._file_handles.keys()):
        self.close(tag)

```

## __enter__

```
__enter__()

```

Context manager entry.

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def __enter__(self):
    """Context manager entry."""
    return self

```

## __exit__

```
__exit__(exc_type, exc_val, exc_tb)

```

Context manager exit - close all files.

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - close all files."""
    self.close_all()

```

## save_batch

```
save_batch(
    samples: StringSampleList, tag: str = "train", batch_idx: int = 0
) -> None

```

Save a batch of samples to files in multiple formats.

This method saves samples in three formats:

1. Pickle (.pkl) - Binary format, always saved, used for loading
1. Raw text (.txt) - Human-readable format with separators (if save_text=True)
1. JSON Lines (.jsonl) - Structured format preserving nested structure (if save_json=True)

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `samples` | `StringSampleList` | List of (problem, solution) pairs in string format | *required* | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | `'train'` | | `batch_idx` | `int` | Batch index for incremental saving. Use 0 for first batch, subsequent batches will append to existing files. | `0` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid or samples contain invalid nested structures |

Examples:

```
>>> # Simple string samples (single problem-solution pairs)
>>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
>>> samples = [
...     ("x^2 + 2*x + 1", "(x + 1)^2"),
...     ("2*x^3 - 3*x^2", "x^2*(2*x - 3)"),
... ]
>>> # Creates: train_data.pkl, train_raw.txt, train_data.jsonl
>>> writer.save_batch(samples, tag="train", batch_idx=0)
>>>
>>> # 1 level nested structure samples (multiple problems/solutions)
>>> samples = [
...     (["x + y", "x - y"], ["2*x", "2*y"]),
...     (["x^2 + y^2", "x^2 - y^2"], ["2*x^2", "2*y^2"]),
... ]
>>> # Text output: "x + y | x - y # 2*x | 2*y"
>>> writer.save_batch(samples, tag="test", batch_idx=0)
>>>
>>> # 2 level nested structure samples (complex nested problems)
>>> samples = [
...     ([["x", "y"], ["z", "w"]], [["x", "z"], ["y", "w"]]),
...     ([["x + y", "x - y"], ["z + w", "z - w"]], [["x + y", "z + w"], ["x - y", "z - w"]]),
... ]
>>> # Text output: "x | y || z | w # x | z || y | w"
>>> writer.save_batch(samples, tag="test", batch_idx=0)
>>>
>>> # Append more samples to existing dataset
>>> more_samples = [
...     ([["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]),
...     ([["e", "f"], ["g", "h"]], [["e", "g"], ["f", "h"]]),
... ]
>>> # Appends to existing files instead of overwriting
>>> writer.save_batch(more_samples, tag="train", batch_idx=1)

```

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def save_batch(
    self,
    samples: StringSampleList,
    tag: str = "train",
    batch_idx: int = 0,
) -> None:
    """
    Save a batch of samples to files in multiple formats.

    This method saves samples in three formats:
    1. Pickle (.pkl) - Binary format, always saved, used for loading
    2. Raw text (.txt) - Human-readable format with separators (if save_text=True)
    3. JSON Lines (.jsonl) - Structured format preserving nested structure (if save_json=True)

    Args:
        samples: List of (problem, solution) pairs in string format
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")
        batch_idx: Batch index for incremental saving. Use 0 for first batch,
                  subsequent batches will append to existing files.

    Raises:
        ValueError: If tag is invalid or samples contain invalid nested structures

    Examples:
        >>> # Simple string samples (single problem-solution pairs)
        >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
        >>> samples = [
        ...     ("x^2 + 2*x + 1", "(x + 1)^2"),
        ...     ("2*x^3 - 3*x^2", "x^2*(2*x - 3)"),
        ... ]
        >>> # Creates: train_data.pkl, train_raw.txt, train_data.jsonl
        >>> writer.save_batch(samples, tag="train", batch_idx=0)
        >>>
        >>> # 1 level nested structure samples (multiple problems/solutions)
        >>> samples = [
        ...     (["x + y", "x - y"], ["2*x", "2*y"]),
        ...     (["x^2 + y^2", "x^2 - y^2"], ["2*x^2", "2*y^2"]),
        ... ]
        >>> # Text output: "x + y | x - y # 2*x | 2*y"
        >>> writer.save_batch(samples, tag="test", batch_idx=0)
        >>>
        >>> # 2 level nested structure samples (complex nested problems)
        >>> samples = [
        ...     ([["x", "y"], ["z", "w"]], [["x", "z"], ["y", "w"]]),
        ...     ([["x + y", "x - y"], ["z + w", "z - w"]], [["x + y", "z + w"], ["x - y", "z - w"]]),
        ... ]
        >>> # Text output: "x | y || z | w # x | z || y | w"
        >>> writer.save_batch(samples, tag="test", batch_idx=0)
        >>>
        >>> # Append more samples to existing dataset
        >>> more_samples = [
        ...     ([["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]),
        ...     ([["e", "f"], ["g", "h"]], [["e", "g"], ["f", "h"]]),
        ... ]
        >>> # Appends to existing files instead of overwriting
        >>> writer.save_batch(more_samples, tag="train", batch_idx=1)
    """
    self._validate_tag(tag)

    # Validate samples
    if not samples:
        self.logger.warning(
            "Empty samples list provided - no files will be created"
        )
        return

    # Check if file handles are open for this tag
    if tag not in self._file_handles:
        # Fallback to old method if file handles are not open
        self._save_batch_legacy(samples, tag, batch_idx)
        return

    # Save binary data (pickle format) - save batch individually
    batch_file = (
        self._file_handles[tag]["batch_dir"]
        / f"batch_{self._file_handles[tag]['batch_count']:06d}.pkl"
    )
    with open(batch_file, "wb") as f:
        pickle.dump(samples, f)
    self._file_handles[tag]["batch_count"] += 1

    # Save raw text data (optional)
    if self.save_text:
        for problem_str, solution_str in samples:
            formatted_line = self._format_sample_strings(problem_str, solution_str)
            self._file_handles[tag]["text"].write(f"{formatted_line}\n")
        self._file_handles[tag]["text"].flush()

    # Save JSON Lines data (optional)
    if self.save_json:
        for problem_str, solution_str in samples:
            json_data = self._get_json_data(problem_str, solution_str)
            json_line = json.dumps(json_data, ensure_ascii=False)
            self._file_handles[tag]["json"].write(f"{json_line}\n")
        self._file_handles[tag]["json"].flush()

```

## save_final_statistics

```
save_final_statistics(statistics: StatisticsDict, tag: str = 'train') -> None

```

Save final overall statistics to YAML file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `statistics` | `StatisticsDict` | Dictionary containing dataset statistics | *required* | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | `'train'` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid |

Note

Statistics are saved in YAML format for human readability. The file is named "{tag}\_stats.yaml" in the dataset directory.

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def save_final_statistics(
    self,
    statistics: StatisticsDict,
    tag: str = "train",
) -> None:
    """
    Save final overall statistics to YAML file.

    Args:
        statistics: Dictionary containing dataset statistics
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Raises:
        ValueError: If tag is invalid

    Note:
        Statistics are saved in YAML format for human readability.
        The file is named "{tag}_stats.yaml" in the dataset directory.
    """
    self._validate_tag(tag)
    dataset_dir = self._create_dataset_dir()

    stats_path = dataset_dir / f"{tag}_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(
            statistics,
            f,
            Dumper=TimedeltaDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=4,
        )

```

## load_dataset

```
load_dataset(tag: str) -> StringSampleList

```

Load dataset from pickle file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Returns:

| Type | Description | | --- | --- | | `StringSampleList` | List of (problem, solution) pairs in string format |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid | | `FileNotFoundError` | If the pickle file doesn't exist |

Examples:

```
>>> samples = writer.load_dataset("train")
>>> print(f"Loaded {len(samples)} samples")
>>> for problem, solution in samples[:3]:
...     print(f"Problem: {problem}, Solution: {solution}")

```

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def load_dataset(self, tag: str) -> StringSampleList:
    """
    Load dataset from pickle file.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Returns:
        List of (problem, solution) pairs in string format

    Raises:
        ValueError: If tag is invalid
        FileNotFoundError: If the pickle file doesn't exist

    Examples:
        >>> samples = writer.load_dataset("train")
        >>> print(f"Loaded {len(samples)} samples")
        >>> for problem, solution in samples[:3]:
        ...     print(f"Problem: {problem}, Solution: {solution}")
    """
    self._validate_tag(tag)
    pickle_path = self.save_dir / f"{tag}_data.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)

```

## load_dataset_jsonl

```
load_dataset_jsonl(tag: str) -> StringSampleList

```

Load dataset from JSON Lines file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tag` | `str` | Dataset tag (e.g., "train", "test", "validation", "dev", "eval") | *required* |

Returns:

| Type | Description | | --- | --- | | `StringSampleList` | List of (problem, solution) pairs in string format |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If tag is invalid | | `FileNotFoundError` | If the JSON Lines file doesn't exist |

Examples:

```
>>> samples = writer.load_dataset_jsonl("train")
>>> print(f"Loaded {len(samples)} samples")
>>> for problem, solution in samples[:3]:
...     print(f"Problem: {problem}, Solution: {solution}")

```

Source code in `src/calt/dataset_generator/sympy/utils/dataset_writer.py`

```
def load_dataset_jsonl(self, tag: str) -> StringSampleList:
    """
    Load dataset from JSON Lines file.

    Args:
        tag: Dataset tag (e.g., "train", "test", "validation", "dev", "eval")

    Returns:
        List of (problem, solution) pairs in string format

    Raises:
        ValueError: If tag is invalid
        FileNotFoundError: If the JSON Lines file doesn't exist

    Examples:
        >>> samples = writer.load_dataset_jsonl("train")
        >>> print(f"Loaded {len(samples)} samples")
        >>> for problem, solution in samples[:3]:
        ...     print(f"Problem: {problem}, Solution: {solution}")
    """
    self._validate_tag(tag)
    jsonl_path = self.save_dir / f"{tag}_data.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSON Lines file not found: {jsonl_path}")

    samples = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                problem = data["problem"]
                solution = data["solution"]
                samples.append((problem, solution))
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    return samples

```

Memory-efficient statistics calculator that uses incremental computation.

This calculator avoids storing all data in memory by computing statistics incrementally as batches are processed using Welford's online algorithm for numerical stability and memory efficiency. All standard deviations are calculated as population standard deviations.

Source code in `src/calt/dataset_generator/sympy/utils/statistics_calculator.py`

```
def __init__(self):
    """Initialize incremental sample statistics calculator."""
    self.runtime_stats = IncrementalStatistics()
    self.sample_stats = {}  # Store aggregated sample statistics by category

```

## update_batch

```
update_batch(
    runtimes: list[float],
    batch_sample_stats: list[dict[str, dict[str, int | float]]],
) -> None

```

Update statistics with a batch of results using Welford's online algorithm.

This method processes each sample individually, updating both runtime statistics and sample-specific statistics incrementally for better control and efficiency.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `runtimes` | `list[float]` | List of runtime values for each sample in the batch | *required* | | `batch_sample_stats` | `list[dict[str, dict[str, int | float]]]` | List of sample statistics dictionaries for the current batch. Each dictionary has the structure: {"category1": {"metric1": value1, ...}, "category2": {"metric1": value1, ...}} Example: [{"problem": {"total_degree": 2, "num_polynomials": 3}, "solution": {"total_degree": 3, "num_polynomials": 3}}, {"problem": {"total_degree": 5, "num_polynomials": 4}, "solution": {"total_degree": 8, "num_polynomials": 4}}, ...] | *required* |

Source code in `src/calt/dataset_generator/sympy/utils/statistics_calculator.py`

```
def update_batch(
    self,
    runtimes: list[float],
    batch_sample_stats: list[dict[str, dict[str, int | float]]],
) -> None:
    """
    Update statistics with a batch of results using Welford's online algorithm.

    This method processes each sample individually, updating both runtime
    statistics and sample-specific statistics incrementally for better
    control and efficiency.

    Args:
        runtimes: List of runtime values for each sample in the batch
        batch_sample_stats: List of sample statistics dictionaries for the current batch.
                           Each dictionary has the structure:
                           {"category1": {"metric1": value1, ...},
                            "category2": {"metric1": value1, ...}}
                           Example:
                           [{"problem": {"total_degree": 2, "num_polynomials": 3},
                             "solution": {"total_degree": 3, "num_polynomials": 3}},
                            {"problem": {"total_degree": 5, "num_polynomials": 4},
                             "solution": {"total_degree": 8, "num_polynomials": 4}},
                            ...]
    """
    # Update runtime statistics
    for runtime in runtimes:
        self.runtime_stats.update(runtime)

    # Update sample statistics
    for stats in batch_sample_stats:
        # Update each numeric sample statistic incrementally
        for category, category_stats in stats.items():
            if isinstance(category_stats, dict):
                # Handle nested structure like {"problem": {...}, "solution": {...}}
                if category not in self.sample_stats:
                    self.sample_stats[category] = {}

                for stat_name, value in category_stats.items():
                    if isinstance(value, (int, float)):
                        if stat_name not in self.sample_stats[category]:
                            self.sample_stats[category][stat_name] = (
                                IncrementalStatistics()
                            )
                        self.sample_stats[category][stat_name].update(float(value))

            elif isinstance(category_stats, (int, float)):
                # Handle flat structure
                if category not in self.sample_stats:
                    self.sample_stats[category] = IncrementalStatistics()
                self.sample_stats[category].update(float(category_stats))

```

## get_overall_statistics

```
get_overall_statistics(total_time: float, num_samples: int) -> dict[str, Any]

```

Get overall statistics.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `total_time` | `float` | Total processing time | *required* | | `num_samples` | `int` | Total number of samples | *required* |

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | Dictionary containing overall statistics with the structure: | | `dict[str, Any]` | { "total_time": float, "num_samples": int, "samples_per_second": float, "generation_time": {"mean": float, "std": float, "min": float, "max": float}, "problem_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...}, "solution_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...} | | `dict[str, Any]` | } |

Source code in `src/calt/dataset_generator/sympy/utils/statistics_calculator.py`

```
def get_overall_statistics(
    self, total_time: float, num_samples: int
) -> dict[str, Any]:
    """
    Get overall statistics.

    Args:
        total_time: Total processing time
        num_samples: Total number of samples

    Returns:
        Dictionary containing overall statistics with the structure:
        {
            "total_time": float,
            "num_samples": int,
            "samples_per_second": float,
            "generation_time": {"mean": float, "std": float, "min": float, "max": float},
            "problem_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...},
            "solution_stats": {"metric1": {"mean": float, "std": float, "min": float, "max": float}, ...}
        }
    """
    runtime_stats = self.runtime_stats.get_statistics()

    overall_stats = {
        "total_time": total_time,
        "num_samples": num_samples,
        "samples_per_second": num_samples / total_time if total_time > 0 else 0.0,
        "generation_time": {
            "mean": runtime_stats["mean"],
            "std": runtime_stats["std"],
            "min": runtime_stats["min"],
            "max": runtime_stats["max"],
        },
    }

    # Add sample statistics by category
    for category, category_stats in self.sample_stats.items():
        if isinstance(category_stats, dict):
            # Handle nested structure like {"problem": {...}, "solution": {...}}
            overall_stats[f"{category}_stats"] = {
                stat_name: stat_calc.get_statistics()
                for stat_name, stat_calc in category_stats.items()
            }
        else:
            # Handle flat structure
            overall_stats[f"{category}_stats"] = category_stats.get_statistics()

    return overall_stats

```

### Sampling

Generator for random polynomials with specific constraints

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `symbols` | `str` | Symbols of polynomial ring | *required* | | `field_str` | `str` | Field of polynomial ring | *required* | | `order` | `str | MonomialOrder` | Order of polynomial ring | *required* | | `max_num_terms` | `int | None` | Maximum number of terms in polynomial. If None, all possible terms are allowed. | `10` | | `max_degree` | `int` | Maximum degree of polynomial | `5` | | `min_degree` | `int` | Minimum degree of polynomial | `0` | | `max_coeff` | `int | None` | Maximum coefficient value | `None` | | `num_bound` | `int | None` | Maximum absolute value of coefficients | `None` | | `degree_sampling` | `str` | How to sample degree ('uniform' or 'fixed') | `'uniform'` | | `term_sampling` | `str` | How to sample number of terms ('uniform' or 'fixed') | `'uniform'` | | `strictly_conditioned` | `bool` | Whether to strictly enforce conditions | `True` | | `nonzero_instance` | `bool` | Whether to enforce non-zero instance | `True` | | `max_attempts` | `int` | Maximum number of attempts to generate a polynomial satisfying conditions | `1000` |

Source code in `src/calt/dataset_generator/sympy/utils/polynomial_sampler.py`

```
def __init__(
    self,
    symbols: str,
    field_str: str,
    order: str | MonomialOrder,
    max_num_terms: int | None = 10,
    max_degree: int = 5,
    min_degree: int = 0,
    degree_sampling: str = "uniform",  # 'uniform' or 'fixed'
    term_sampling: str = "uniform",  # 'uniform' or 'fixed'
    max_coeff: int | None = None,  # Used for RR and ZZ
    num_bound: int | None = None,  # Used for QQ
    strictly_conditioned: bool = True,
    nonzero_instance: bool = True,
    max_attempts: int = 1000,
) -> None:
    """
    Initialize polynomial sampler

    Args:
        symbols: Symbols of polynomial ring
        field_str: Field of polynomial ring
        order: Order of polynomial ring
        max_num_terms: Maximum number of terms in polynomial. If None, all possible terms are allowed.
        max_degree: Maximum degree of polynomial
        min_degree: Minimum degree of polynomial
        max_coeff: Maximum coefficient value
        num_bound: Maximum absolute value of coefficients
        degree_sampling: How to sample degree ('uniform' or 'fixed')
        term_sampling: How to sample number of terms ('uniform' or 'fixed')
        strictly_conditioned: Whether to strictly enforce conditions
        nonzero_instance: Whether to enforce non-zero instance
        max_attempts: Maximum number of attempts to generate a polynomial satisfying conditions
    """

    self.symbols = symbols
    self.field_str = field_str
    self.order = order
    self.max_num_terms = max_num_terms
    self.max_degree = max_degree
    self.min_degree = min_degree
    self.max_coeff = max_coeff
    self.num_bound = num_bound
    self.degree_sampling = degree_sampling
    self.term_sampling = term_sampling
    self.strictly_conditioned = strictly_conditioned
    self.nonzero_instance = nonzero_instance
    self.max_attempts = max_attempts
    self.single_poly_sampler = SinglePolynomialSampler()

```

## get_field

```
get_field() -> Domain

```

Convert field_str to actual sympy domain object

Source code in `src/calt/dataset_generator/sympy/utils/polynomial_sampler.py`

```
def get_field(self) -> Domain:
    """Convert field_str to actual sympy domain object"""
    # Standard field mapping
    standard_fields = {"QQ": QQ, "RR": RR, "ZZ": ZZ}
    if self.field_str in standard_fields:
        return standard_fields[self.field_str]

    # Finite field handling
    if not self.field_str.startswith("GF"):
        raise ValueError(f"Unsupported field: {self.field_str}")

    try:
        # Extract field size based on format
        p = int(
            self.field_str[3:-1]
            if self.field_str.startswith("GF(")
            else self.field_str[2:]
        )

        if p <= 1:
            raise ValueError(f"Field size must be greater than 1: {p}")
        return GF(p)
    except ValueError as e:
        raise ValueError(f"Unsupported field: {self.field_str}") from e

```

## get_ring

```
get_ring() -> PolyRing

```

Generate polynomial ring

Returns:

| Name | Type | Description | | --- | --- | --- | | `PolyRing` | `PolyRing` | Generated polynomial ring |

Source code in `src/calt/dataset_generator/sympy/utils/polynomial_sampler.py`

```
def get_ring(self) -> PolyRing:
    """
    Generate polynomial ring

    Returns:
        PolyRing: Generated polynomial ring
    """

    R, *gens = ring(self.symbols, self.get_field(), self.order)
    return R

```

## sample

```
sample(
    num_samples: int = 1,
    size: tuple[int, int] | None = None,
    density: float = 1.0,
    matrix_type: str | None = None,
) -> list[PolyElement] | list[np.ndarray]

```

Generate random polynomial samples

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `num_samples` | `int` | Number of samples to generate | `1` | | `size` | `tuple[int, int] | None` | If provided, generate matrix of polynomials with given size | `None` | | `density` | `float` | Probability of non-zero entries in matrix | `1.0` | | `matrix_type` | `str | None` | Special matrix type (e.g., 'unimodular_upper_triangular') | `None` |

Returns:

| Type | Description | | --- | --- | | `list[PolyElement] | list[ndarray]` | List of polynomials or polynomial matrices |

Source code in `src/calt/dataset_generator/sympy/utils/polynomial_sampler.py`

```
def sample(
    self,
    num_samples: int = 1,
    size: tuple[int, int] | None = None,
    density: float = 1.0,
    matrix_type: str | None = None,
) -> list[PolyElement] | list[np.ndarray]:
    """
    Generate random polynomial samples

    Args:
        num_samples: Number of samples to generate
        size: If provided, generate matrix of polynomials with given size
        density: Probability of non-zero entries in matrix
        matrix_type: Special matrix type (e.g., 'unimodular_upper_triangular')

    Returns:
        List of polynomials or polynomial matrices
    """
    if size is not None:
        return [
            self._sample_matrix(size, density, matrix_type)
            for _ in range(num_samples)
        ]
    else:
        return [self._sample_polynomial() for _ in range(num_samples)]

```

## total_degree

```
total_degree(poly: PolyElement) -> int

```

Compute total degree of a polynomial

Source code in `src/calt/dataset_generator/sympy/utils/polynomial_sampler.py`

```
def total_degree(self, poly: PolyElement) -> int:
    """Compute total degree of a polynomial"""
    if poly.is_zero:
        return 0
    else:
        return max(sum(monom) for monom in poly.monoms())

```

Sampler for single polynomial with specific constraints

## random_coeff

```
random_coeff(field: Domain, non_zero: bool = False, **kwargs) -> Any

```

Generate a random coefficient in the given field.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `field` | `Domain` | The coefficient field (e.g., ZZ, QQ, RR, GF) | *required* | | `non_zero` | `bool` | If True, ensure the coefficient is non-zero | `False` | | `**kwargs` | | Additional parameters for coefficient generation - min: minimum value (default: -10) - max: maximum value (default: 10) - num_bound: bound for numerator and denominator in QQ (default: 10) | `{}` |

Returns:

| Type | Description | | --- | --- | | `Any` | Random coefficient in the specified field |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If parameter ranges are invalid or non_zero cannot be satisfied | | `NotImplementedError` | If the field is not supported |

Source code in `src/calt/dataset_generator/sympy/utils/single_polynomial_sampler.py`

```
def random_coeff(self, field: Domain, non_zero: bool = False, **kwargs) -> Any:
    """
    Generate a random coefficient in the given field.

    Args:
        field: The coefficient field (e.g., ZZ, QQ, RR, GF)
        non_zero: If True, ensure the coefficient is non-zero
        **kwargs: Additional parameters for coefficient generation
            - min: minimum value (default: -10)
            - max: maximum value (default: 10)
            - num_bound: bound for numerator and denominator in QQ (default: 10)

    Returns:
        Random coefficient in the specified field

    Raises:
        ValueError: If parameter ranges are invalid or non_zero cannot be satisfied
        NotImplementedError: If the field is not supported
    """

    # Integer coefficient
    if field == ZZ:
        a = kwargs.get("min", -10)
        b = kwargs.get("max", 10)

        if a > b:
            raise ValueError("min must be <= max")

        if non_zero and a == 0 and b == 0:
            raise ValueError("Cannot generate non-zero ZZ with min=0 and max=0")

        # Define a generator function that returns a random ZZ in [a, b]
        def gen_int():
            return ZZ(random.randint(a, b))

        return self._pick_random_until_nonzero(gen_int, non_zero)

    # Real number coefficient
    elif field == RR:
        a = kwargs.get("min", -10.0)
        b = kwargs.get("max", 10.0)

        if a > b:
            raise ValueError("min must be <= max")

        if non_zero and a == 0.0 and b == 0.0:
            raise ValueError("Cannot generate non-zero RR with min=0.0 and max=0.0")

        # Define a generator function that returns a random RR in [a, b]
        def gen_real():
            return RR(random.uniform(a, b))

        return self._pick_random_until_nonzero(gen_real, non_zero)

    # Rational number coefficient
    elif field == QQ:
        num_bound = kwargs.get("num_bound", 10)

        if num_bound <= 0:
            raise ValueError("num_bound must be > 0")

        # Define a generator function that returns a random QQ with numerator in [-num_bound, num_bound] and denominator in [1, num_bound]
        def gen_rat():
            numerator = random.randint(-num_bound, num_bound)
            denominator = random.randint(1, num_bound)
            return QQ(numerator, denominator)

        return self._pick_random_until_nonzero(gen_rat, non_zero)

    # Finite field
    elif field.is_FiniteField:
        p = field.characteristic()

        if non_zero and p == 1:
            raise ValueError(
                "Cannot generate non-zero finite field coefficient with characteristic 1"
            )

        # Define a generator function that returns a random field element in GF(p)
        def gen_gf():
            return field(random.randint(0, p - 1))

        return self._pick_random_until_nonzero(gen_gf, non_zero)

    else:
        raise NotImplementedError(
            f"Random coefficient generation not implemented for field {field}"
        )

```

## random_element

```
random_element(
    R: PolyRing,
    degree: int = 2,
    terms: int | None = None,
    choose_degree: bool = False,
    non_zero_coeff: bool = False,
    **kwargs,
) -> PolyElement

```

Return a random polynomial of at most the specified degree and at most the specified number of terms.

First monomials are chosen uniformly random from the set of all possible monomials of degree up to the specified degree (inclusive). This means that it is more likely that a monomial of the specified degree appears than a monomial of degree (specified degree - 1) because the former class is bigger.

Exactly the specified number of distinct monomials are chosen this way and each one gets a random coefficient (possibly zero) from the base ring assigned.

The returned polynomial is the sum of this list of terms.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `R` | `PolyRing` | Polynomial ring | *required* | | `degree` | `int` | Maximum degree of the polynomial | `2` | | `terms` | `int | None` | Number of terms in the polynomial | `None` | | `choose_degree` | `bool` | Whether to choose degree randomly first | `False` | | `non_zero_coeff` | `bool` | If True, ensure all coefficients are non-zero | `False` | | `**kwargs` | | Additional parameters for coefficient generation - min: minimum value (default: -10) - max: maximum value (default: 10) - num_bound: bound for numerator and denominator in QQ (default: 10) | `{}` |

Returns:

| Type | Description | | --- | --- | | `PolyElement` | Random polynomial in the given ring |

Source code in `src/calt/dataset_generator/sympy/utils/single_polynomial_sampler.py`

```
def random_element(
    self,
    R: PolyRing,
    degree: int = 2,
    terms: int | None = None,
    choose_degree: bool = False,
    non_zero_coeff: bool = False,
    **kwargs,
) -> PolyElement:
    """
    Return a random polynomial of at most the specified degree and at most the specified number of terms.

    First monomials are chosen uniformly random from the set of all
    possible monomials of degree up to the specified degree (inclusive). This means
    that it is more likely that a monomial of the specified degree appears than
    a monomial of degree (specified degree - 1) because the former class is bigger.

    Exactly the specified number of distinct monomials are chosen this way and each one gets
    a random coefficient (possibly zero) from the base ring assigned.

    The returned polynomial is the sum of this list of terms.

    Args:
        R: Polynomial ring
        degree: Maximum degree of the polynomial
        terms: Number of terms in the polynomial
        choose_degree: Whether to choose degree randomly first
        non_zero_coeff: If True, ensure all coefficients are non-zero
        **kwargs: Additional parameters for coefficient generation
            - min: minimum value (default: -10)
            - max: maximum value (default: 10)
            - num_bound: bound for numerator and denominator in QQ (default: 10)


    Returns:
        Random polynomial in the given ring
    """
    field = R.domain
    n = R.ngens

    counts, total = self._precomp_counts(n, degree)

    if terms is not None and terms < 0:
        raise ValueError("terms must be >= 0")
    if degree < 0:
        raise ValueError("degree must be >= 0")

    # special cases
    if terms == 0:
        return R.zero
    if degree == 0:
        return R(self.random_coeff(field=field, non_zero=non_zero_coeff, **kwargs))

    # adjust terms
    if terms is None:
        terms = min(5, total)
    else:
        terms = min(terms, total)

    # total is 0. Just return
    if total == 0:
        return R.zero
    elif terms < total / 2:
        # we choose random monomials if t < total/2 because then we
        # expect the algorithm to be faster than generating all
        # monomials and picking a random index from the list. if t ==
        # total/2 we expect every second random monomial to be a
        # double such that our runtime is doubled in the worst case.
        M: set[tuple[int, ...]] = set()
        if not choose_degree:
            while terms:
                m = self._random_monomial_upto_degree_uniform(
                    n, degree, counts, total
                )
                if m not in M:
                    M.add(m)
                    terms -= 1
        else:
            while terms:
                m = self._random_monomial_upto_degree_class(n, degree)
                if m not in M:
                    M.add(m)
                    terms -= 1
    elif terms <= total:
        # generate a list of all monomials and choose among them
        if not choose_degree:
            M = sum(
                [list(self._integer_vectors(_d, n)) for _d in range(degree + 1)], []
            )
            # we throw away those we don't need
            for mi in range(total - terms):
                M.pop(random.randint(0, len(M) - 1))
            M = [tuple(m) for m in M]
        else:
            M = [list(self._integer_vectors(_d, n)) for _d in range(degree + 1)]
            Mbar = []
            for mi in range(terms):
                # choose degree 'd' and monomial 'm' at random
                d = random.randint(0, len(M) - 1)
                m = random.randint(0, len(M[d]) - 1)
                Mbar.append(M[d].pop(m))  # remove and insert
                if len(M[d]) == 0:
                    M.pop(d)  # bookkeeping
            M = [tuple(m) for m in Mbar]

    # Generate random coefficients
    C = [
        self.random_coeff(field=field, non_zero=non_zero_coeff, **kwargs)
        for _ in range(len(M))
    ]

    # Create the polynomial using from_dict
    return R.from_dict(dict(zip(M, C)))

```
