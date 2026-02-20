# SymPy backend

When using `backend="sympy"`, the following classes are used for generation and sampling. You can also use them directly without [DatasetPipeline](../dataset_generator/).

See [Dataset Generator (Overview)](../dataset_generator/) for the pipeline and `data.yaml` configuration.

## DatasetGenerator

```
DatasetGenerator(
    backend: str = "multiprocessing",
    n_jobs: int = -1,
    verbose: bool = True,
    root_seed: int = 42,
)
```

Base class for instance generators

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `backend` | `str` | Backend for parallel processing | `'multiprocessing'` | | `n_jobs` | `int` | Number of parallel jobs (-1 for all cores) | `-1` | | `verbose` | `bool` | Whether to display progress information | `True` | | `root_seed` | `int` | Root seed for reproducibility | `42` |

Source code in `src/calt/dataset/sympy/dataset_generator.py`

```
def __init__(
    self,
    backend: str = "multiprocessing",
    n_jobs: int = -1,
    verbose: bool = True,
    root_seed: int = 42,
):
    """
    Initialize instance generator.

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

### run

```
run(
    dataset_sizes: dict[str, int],
    instance_generator: Callable,
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
- Multiple output formats (text, JSON Lines) via DatasetWriter

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `dataset_sizes` | `dict[str, int]` | Dictionary mapping dataset names to number of samples. Any string can be used as dataset name (e.g., "train", "test", "validation"). Duplicate names are not allowed. Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000} | *required* | | `instance_generator` | `Callable` | Function that generates (problem, answer) pair given a seed. Must accept a single integer seed parameter. | *required* | | `statistics_calculator` | `Callable | None` | Optional function to calculate sample-specific statistics. Must accept (problem, answer) and return dict or None. | `None` | | `dataset_writer` | `DatasetWriter | None` | DatasetWriter object for saving datasets to files. If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters. | `None` | | `batch_size` | `int` | Number of samples to process in each batch. Larger batches use more memory but may be more efficient for I/O operations. | `100000` | | `save_dir` | `str | None` | Base directory for saving datasets. Used only if dataset_writer is None. If None, uses current working directory. | `None` | | `save_text` | `bool` | Whether to save raw text files. Used only if dataset_writer is None. Text files use "#" as separator between problem and answer. | `True` | | `save_json` | `bool` | Whether to save JSON Lines files. Used only if dataset_writer is None. JSON Lines files preserve the original nested structure format. | `True` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If dataset_sizes is invalid or instance_generator is None | | `Exception` | If parallel processing fails |

Note

- Each sample gets a unique seed for reproducibility
- Progress is logged if verbose=True (set in **init**)
- Memory usage scales with batch_size, not total dataset size
- Statistics are calculated incrementally to handle large datasets
- If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

Examples:

```
>>> # Define instance generator function
>>> def instance_generator(seed):
...     import random
...     random.seed(seed)
...     # Generate random polynomial problem
...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
...     answer = sum(problem)
...     return problem, answer
>>>
>>> # Initialize dataset generator
>>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
>>>
>>> # Method 1: Automatic DatasetWriter creation
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
...     instance_generator=instance_generator,
...     save_dir="./datasets",
...     save_text=True,
...     save_json=True,
...     batch_size=100
... )
>>>
>>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
>>> from calt.dataset.sympy import DatasetWriter
>>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
>>> generator.run(
...     dataset_sizes={"train": 10000, "test": 1000},
...     instance_generator=instance_generator,
...     dataset_writer=writer,
...     batch_size=100
... )
>>>
>>> # Method 3: Generate datasets separately (if needed)
>>> generator.run(
...     dataset_sizes={"train": 10000},
...     instance_generator=instance_generator,
...     save_dir="./datasets",
...     batch_size=100
... )
>>> generator.run(
...     dataset_sizes={"test": 1000, "validation": 500},
...     instance_generator=instance_generator,
...     save_dir="./datasets",
...     batch_size=100
... )
```

Source code in `src/calt/dataset/sympy/dataset_generator.py`

```
def run(
    self,
    dataset_sizes: dict[str, int],
    instance_generator: Callable,
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
    - Multiple output formats (text, JSON Lines) via DatasetWriter

    Args:
        dataset_sizes: Dictionary mapping dataset names to number of samples.
                      Any string can be used as dataset name (e.g., "train", "test", "validation").
                      Duplicate names are not allowed.
                      Example: {"train": 100000, "test": 1000} or {"train": 100000, "validation": 5000}
        instance_generator: Function that generates (problem, answer) pair given a seed.
                         Must accept a single integer seed parameter.
        statistics_calculator: Optional function to calculate sample-specific statistics.
                             Must accept (problem, answer) and return dict or None.
        dataset_writer: DatasetWriter object for saving datasets to files.
                      If None, a new DatasetWriter will be created using save_dir, save_text, and save_json parameters.
        batch_size: Number of samples to process in each batch. Larger batches
                   use more memory but may be more efficient for I/O operations.
        save_dir: Base directory for saving datasets. Used only if dataset_writer is None.
                 If None, uses current working directory.
        save_text: Whether to save raw text files. Used only if dataset_writer is None.
                  Text files use "#" as separator between problem and answer.
        save_json: Whether to save JSON Lines files. Used only if dataset_writer is None.
                  JSON Lines files preserve the original nested structure format.

    Raises:
        ValueError: If dataset_sizes is invalid or instance_generator is None
        Exception: If parallel processing fails

    Note:
        - Each sample gets a unique seed for reproducibility
        - Progress is logged if verbose=True (set in __init__)
        - Memory usage scales with batch_size, not total dataset size
        - Statistics are calculated incrementally to handle large datasets
        - If dataset_writer is provided, save_dir, save_text, and save_json parameters are ignored

    Examples:
        >>> # Define instance generator function
        >>> def instance_generator(seed):
        ...     import random
        ...     random.seed(seed)
        ...     # Generate random polynomial problem
        ...     problem = [random.randint(1, 1000) for _ in range(random.randint(1, 10))]
        ...     answer = sum(problem)
        ...     return problem, answer
        >>>
        >>> # Initialize dataset generator
        >>> generator = DatasetGenerator(n_jobs=-1, verbose=True)
        >>>
        >>> # Method 1: Automatic DatasetWriter creation
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000, "validation": 500},
        ...     instance_generator=instance_generator,
        ...     save_dir="./datasets",
        ...     save_text=True,
        ...     save_json=True,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 2: Manual DatasetWriter creation (for advanced use cases)
        >>> from calt.dataset.sympy import DatasetWriter
        >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000, "test": 1000},
        ...     instance_generator=instance_generator,
        ...     dataset_writer=writer,
        ...     batch_size=100
        ... )
        >>>
        >>> # Method 3: Generate datasets separately (if needed)
        >>> generator.run(
        ...     dataset_sizes={"train": 10000},
        ...     instance_generator=instance_generator,
        ...     save_dir="./datasets",
        ...     batch_size=100
        ... )
        >>> generator.run(
        ...     dataset_sizes={"test": 1000, "validation": 500},
        ...     instance_generator=instance_generator,
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
        "instance_generator": instance_generator,
        "statistics_calculator": statistics_calculator,
        "dataset_writer": dataset_writer,
        "batch_size": batch_size,
    }

    # Validate dataset_sizes
    if not isinstance(dataset_sizes, dict):
        raise ValueError("dataset_sizes must be a dictionary")

    if not dataset_sizes:
        raise ValueError("dataset_sizes cannot be empty")

    if instance_generator is None:
        raise ValueError("instance_generator must be provided")

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

## PolynomialSampler

```
PolynomialSampler(
    symbols: str,
    field_str: str,
    order: str | MonomialOrder = "grevlex",
    max_num_terms: int | None = 10,
    max_degree: int = 5,
    min_degree: int = 0,
    degree_sampling: str = "uniform",
    term_sampling: str = "uniform",
    max_coeff: int | None = None,
    num_bound: int | None = None,
    strictly_conditioned: bool = True,
    nonzero_instance: bool = True,
    max_attempts: int = 1000,
)
```

Generator for random polynomials with specific constraints (SymPy).

The sampler builds polynomials by choosing a target degree and number of terms (within min/max bounds), then uses :class:`SinglePolynomialSampler` to select that many distinct monomials and assign random coefficients from the base ring. Ring is specified by symbols, field_str, and order.

#### Behavior summary

**degree_sampling** controls how monomial degrees are chosen (passed as `choose_degree` to the internal sampler):

- `'uniform'`: For each term, a degree in [min_degree, max_degree] is chosen uniformly at random, then a monomial of that degree is chosen. The resulting polynomial's degree distribution is more uniform over the range.
- `'fixed'`: Monomials are chosen uniformly from all monomials of degree at most max_degree. The polynomial tends to have total degree equal to max_degree.

**Degree and number of terms**: Every returned polynomial has total degree >= min_degree. The guarantees on total degree and number of terms depend on `strictly_conditioned` and `nonzero_instance`; see the constructor parameters for details.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `symbols` | `str` | Variable names for the polynomial ring. | *required* | | `field_str` | `str` | Base ring specifier: "QQ", "RR", "ZZ", or "GF(p)" for a prime finite field. | *required* | | `order` | `str | MonomialOrder` | Term order of the ring, e.g. "grevlex". | `'grevlex'` | | `max_num_terms` | `int | None` | Upper bound on number of terms. If None, all monomials of the chosen degree are allowed. | `10` | | `max_degree` | `int` | Maximum total degree of the polynomial. | `5` | | `min_degree` | `int` | Minimum total degree; every returned polynomial has total degree >= min_degree. | `0` | | `max_coeff` | `int | None` | Bound on coefficient absolute value for RR and ZZ. | `None` | | `num_bound` | `int | None` | Bound on numerator/denominator absolute value for QQ. | `None` | | `degree_sampling` | `str` | 'uniform' or 'fixed'; see class docstring (Behavior summary). | `'uniform'` | | `term_sampling` | `str` | 'uniform': number of terms chosen uniformly in [1, max_terms] (max_terms bounded by max_num_terms); 'fixed': use max_terms. | `'uniform'` | | `strictly_conditioned` | `bool` | Controls when a generated polynomial is accepted. If True: Return only when total degree equals the degree selected for this sample and number of terms equals the number of terms selected for this sample. (Those values are chosen by degree_sampling and term_sampling; degree is in [min_degree, max_degree], and number of terms is at most max_num_terms.) RuntimeError is raised if no success within max_attempts. If False: Return the first polynomial with total degree >= min_degree and (if nonzero_instance) non-zero. Number of terms may be less than the chosen value when nonzero_instance is False. | `True` | | `nonzero_instance` | `bool` | If True, the zero polynomial is never returned and all coefficients are non-zero (predictable number of terms). If False, coefficients may be zero. | `True` | | `max_attempts` | `int` | Maximum trials per polynomial when strictly_conditioned is True; RuntimeError is raised if no success. | `1000` |

Source code in `src/calt/dataset/sympy/utils/polynomial_sampler.py`

```
def __init__(
    self,
    symbols: str,
    field_str: str,
    order: str | MonomialOrder = "grevlex",
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
    Initialize polynomial sampler.

    Args:
        symbols: Variable names for the polynomial ring.
        field_str: Base ring specifier: "QQ", "RR", "ZZ", or "GF(p)"
            for a prime finite field.
        order: Term order of the ring, e.g. "grevlex".
        max_num_terms: Upper bound on number of terms. If None, all
            monomials of the chosen degree are allowed.
        max_degree: Maximum total degree of the polynomial.
        min_degree: Minimum total degree; every returned polynomial
            has total degree >= min_degree.
        max_coeff: Bound on coefficient absolute value for RR and ZZ.
        num_bound: Bound on numerator/denominator absolute value
            for QQ.
        degree_sampling: ``'uniform'`` or ``'fixed'``; see class
            docstring (Behavior summary).
        term_sampling: ``'uniform'``: number of terms chosen uniformly
            in [1, max_terms] (max_terms bounded by max_num_terms);
            ``'fixed'``: use max_terms.
        strictly_conditioned: Controls when a generated polynomial
            is accepted.

            - If True:
                - Return only when total degree equals the degree
                  selected for this sample and number of terms
                  equals the number of terms selected for this
                  sample. (Those values are chosen by
                  degree_sampling and term_sampling; degree is in
                  [min_degree, max_degree], and number of terms is
                  at most max_num_terms.)
                - RuntimeError is raised if no success within
                  max_attempts.
            - If False:
                - Return the first polynomial with total degree >=
                  min_degree and (if nonzero_instance) non-zero.
                - Number of terms may be less than the chosen value
                  when nonzero_instance is False.
        nonzero_instance: If True, the zero polynomial is never
            returned and all coefficients are non-zero (predictable
            number of terms). If False, coefficients may be zero.
        max_attempts: Maximum trials per polynomial when
            strictly_conditioned is True; RuntimeError is raised if
            no success.
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

### get_field

```
get_field() -> Domain
```

Return the SymPy domain for field_str (QQ, RR, ZZ, or GF(p)).

Source code in `src/calt/dataset/sympy/utils/polynomial_sampler.py`

```
def get_field(self) -> Domain:
    """Return the SymPy domain for field_str (QQ, RR, ZZ, or GF(p))."""
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

### get_ring

```
get_ring() -> PolyRing
```

Return the polynomial ring (PolyRing) for the configured symbols, field, and order.

Source code in `src/calt/dataset/sympy/utils/polynomial_sampler.py`

```
def get_ring(self) -> PolyRing:
    """Return the polynomial ring (PolyRing) for the configured symbols, field, and order."""

    R, *gens = ring(self.symbols, self.get_field(), self.order)
    return R
```

### sample

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

| Type | Description | | --- | --- | | `list[PolyElement] | list[ndarray]` | List of polynomials, or list of polynomial matrices when | | `list[PolyElement] | list[ndarray]` | size is provided. |

Source code in `src/calt/dataset/sympy/utils/polynomial_sampler.py`

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
        List of polynomials, or list of polynomial matrices when
        size is provided.
    """
    if size is not None:
        return [
            self._sample_matrix(size, density, matrix_type)
            for _ in range(num_samples)
        ]
    else:
        return [self._sample_polynomial() for _ in range(num_samples)]
```

### total_degree

```
total_degree(poly: PolyElement) -> int
```

Return the total degree of the polynomial.

Source code in `src/calt/dataset/sympy/utils/polynomial_sampler.py`

```
def total_degree(self, poly: PolyElement) -> int:
    """Return the total degree of the polynomial."""
    if poly.is_zero:
        return 0
    else:
        return max(sum(monom) for monom in poly.monoms())
```
