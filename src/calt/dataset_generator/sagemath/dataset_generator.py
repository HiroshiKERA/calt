import hashlib
import logging
import math
from time import perf_counter
from typing import Any, Callable

from joblib import Parallel, delayed

from .utils.dataset_writer import DatasetWriter
from .utils.statistics_calculator import MemoryEfficientStatisticsCalculator

# Type aliases for better readability
ProblemOrSolution = Any | list[Any] | list[list[Any]] | list[Any | list[Any]]
"""Type alias for problems and solutions in their original format.
Supports single values, simple lists, nested lists, and mixed structures.

Polynomial examples:
    - Single: R("x^2 + 2*x + 1")  # where R is a polynomial ring like R = PolynomialRing(QQ, 'x,y')
    - Simple list: [R("x + y"), R("x - y")]
    - Nested list: [[R("x^2"), R("y^2")], [R("z^2"), R("w^2")]]
    - Mixed: [[R("x"), R("y")], R("z")]
    
    # Common SageMath polynomial creation methods:
    # R = PolynomialRing(QQ, 'x,y,z')  # Create polynomial ring
    # p = R("x^2 + y^2")               # From string
    # p = x^2 + y^2                    # Direct construction (if variables are defined)
    # p = R([1, 0, 1])                 # From coefficient list

Arithmetic examples:
    - Single: 2
    - Simple list: [2, 3, 5, 7]
    - Nested list: [[2, 3], [5, 7]]
    - Mixed: [2, [3, 5]]
"""

StatisticsDict = dict[str, dict[str, int | float]]
"""Type alias for statistics dictionary containing nested metrics.
Example: {"runtime": {"mean": 0.5, "std": 0.1}, "complexity": {"max": 10, "min": 1}}"""

StringProblemOrSolution = str | list[str] | list[list[str]] | list[str | list[str]]
"""Type alias for string-formatted problems and solutions.
Supports single strings, simple lists, nested lists, and mixed structures.
Maximum nesting depth is 2 levels.

Polynomial examples:
    - Single: "x^2 + 2*x + 1"
    - Simple list: ["x + y", "x - y"]
    - Nested list: [["x^2", "y^2"], ["z^2", "w^2"]]
    - Mixed: [["x", "y"], "z"] 

Arithmetic examples:
    - Single: "2"
    - Simple list: ["2", "3", "5", "7"]
    - Nested list: [["2", "3"], ["5", "7"]]
    - Mixed: ["2", ["3", "5"]]
"""


def setup_logging():
    """Setup logging configuration for the application."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Only configure if no handlers exist
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        # Update existing handlers to use our format
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(logging.Formatter("%(message)s"))


def _worker_init():
    setup_logging()


# Setup logging for this module
setup_logging()
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Base class for problem generators"""

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

    def _generate_seed(self, sample_index: int, tag: str) -> int:
        """
        Generate a unique seed value for each job using SHA-256 hash.
        Uses 16 bytes (128 bits) of the hash to ensure extremely low collision probability.

        Args:
            sample_index: Sample identifier (global sample index, independent of batch size)
            tag: Dataset tag (e.g.,"train", "test", "eval")

        Returns:
            Integer seed value (128 bits)
        """
        # Create a unique string for this job (batch-independent)
        seed_str = f"{self.root_seed}_{tag}_{sample_index}"
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(seed_str.encode())
        # Convert first 16 bytes to integer (128 bits) for better collision resistance
        return int.from_bytes(hash_obj.digest()[:16], byteorder="big")

    def _convert_nested_structure(
        self, obj: ProblemOrSolution
    ) -> StringProblemOrSolution:
        """
        Convert nested structure (problem or solution) to string format.
        Handles both simple values, lists, and mixed nested lists.

        Args:
            obj: Object to convert (can be single value, list, or mixed nested list)

        Returns:
            Converted object with strings
        """
        if isinstance(obj, list):
            # Check if this list contains any nested lists
            has_nested_lists = any(isinstance(item, list) for item in obj)

            if has_nested_lists:
                # Mixed structure: handle each item appropriately
                result = []
                for item in obj:
                    if isinstance(item, list):
                        # Inner list: convert each item to string
                        result.append([str(subitem) for subitem in item])
                    else:
                        # Single item: convert to string
                        result.append(str(item))
                return result
            else:
                # Simple list: convert each item to string
                return [str(item) for item in obj]
        else:
            # Single value: convert to string
            return str(obj)

    def _generate_sample(
        self,
        sample_index: int,
        tag: str,
        problem_generator: Callable,
        statistics_calculator: Callable | None = None,
    ) -> tuple[
        StringProblemOrSolution, StringProblemOrSolution, StatisticsDict | None, float
    ]:
        """Generate a single sample using the provided problem generator."""
        # Generate a unique seed for this job
        seed = self._generate_seed(sample_index, tag)

        start_time = perf_counter()
        problem, solution = problem_generator(seed)
        run_time = perf_counter() - start_time

        if statistics_calculator is not None:
            sample_stats = statistics_calculator(problem, solution)
        else:
            sample_stats = None

        # Convert problem and solution to string format, handling nested structures
        problem_str = self._convert_nested_structure(problem)
        solution_str = self._convert_nested_structure(solution)

        return problem_str, solution_str, sample_stats, run_time

    def _generate_dataset(
        self,
        tag: str,
        num_samples: int,
        problem_generator: Callable,
        statistics_calculator: Callable | None = None,
        dataset_writer: DatasetWriter | None = None,
        batch_size: int = 100000,
    ):
        """Generate a single dataset with parallel processing and batch writing."""
        start_time = perf_counter()

        # Initialize memory-efficient statistics calculator
        incremental_stats = MemoryEfficientStatisticsCalculator()

        # Validate batch size
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        # Calculate number of batches
        num_batches = math.ceil(num_samples / batch_size)

        self.logger.info(
            f"---------------------------------- {tag} ----------------------------------"
        )
        self.logger.info(
            f"Dataset size: {num_samples} samples  (Batch size: {batch_size})\n"
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start

            if self.verbose:
                self.logger.info(f"--- Batch {batch_idx + 1}/{num_batches} ---")
                self.logger.info(
                    f"Processing samples {batch_start + 1}-{batch_end} (size: {current_batch_size})"
                )
                self.logger.info("Starting parallel processing...")

            # Generate samples for current batch in parallel using joblib
            try:
                results = Parallel(
                    n_jobs=self.n_jobs,
                    backend=self.backend,
                    verbose=self.verbose,
                    initializer=_worker_init,
                )(
                    delayed(self._generate_sample)(
                        batch_start + i,
                        tag,
                        problem_generator,
                        statistics_calculator,
                    )
                    for i in range(current_batch_size)
                )
            except (MemoryError, OSError) as e:
                self.logger.error(f"System error in batch {batch_idx + 1}: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error in batch {batch_idx + 1}: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                raise

            # Unzip the results for current batch
            problem_strs, solution_strs, sample_stats, run_times = zip(*results)

            if self.verbose:
                self.logger.info("Parallel processing completed")

            # Store batch results for statistics only
            batch_samples = list(zip(problem_strs, solution_strs))

            # Update statistics incrementally
            incremental_stats.update_batch(
                run_times, sample_stats if sample_stats[0] is not None else []
            )

            # Write batch to file if dataset_writer is provided
            if dataset_writer is not None:
                dataset_writer.save_batch(
                    samples=batch_samples, tag=tag, batch_idx=batch_idx
                )

                if self.verbose:
                    self.logger.info(f"Batch {batch_idx + 1} saved to file")

            # Clear batch data from memory to prevent memory buildup
            del batch_samples, problem_strs, solution_strs, sample_stats, run_times

            if self.verbose:
                self.logger.info(f"Batch {batch_idx + 1}/{num_batches} completed")
                self.logger.info("")

        # Calculate overall statistics from incremental data
        total_time = perf_counter() - start_time

        # Always use memory-efficient statistics calculator for overall statistics
        overall_stats = incremental_stats.get_overall_statistics(
            total_time, num_samples
        )

        # Save final overall statistics if dataset_writer is provided
        if dataset_writer is not None:
            dataset_writer.save_final_statistics(statistics=overall_stats, tag=tag)
            self.logger.info(f"Overall statistics saved for {tag} dataset")

        self.logger.info(f"Total time: {overall_stats['total_time']:.2f} seconds\n\n")

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
