from typing import Any, Callable
from joblib import Parallel, delayed
from time import time
import hashlib
import re
import logging
from sympy.polys.rings import PolyElement
from .utils.dataset_writer import DatasetWriter
from .utils.statistics_calculator import MemoryEfficientStatisticsCalculator


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

    def _generate_seed(self, job_id: int, tag: str, batch_idx: int = 0) -> int:
        """
        Generate a unique seed value for each job using SHA-256 hash.
        Uses 16 bytes (128 bits) of the hash to ensure extremely low collision probability.

        Args:
            job_id: Job identifier
            tag: Dataset tag ("train" or "test")
            batch_idx: Batch index for additional uniqueness

        Returns:
            Integer seed value (128 bits)
        """
        # Create a unique string for this job including batch index
        seed_str = f"{self.root_seed}_{tag}_{batch_idx}_{job_id}"
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(seed_str.encode())
        # Convert first 16 bytes to integer (128 bits) for better collision resistance
        return int.from_bytes(hash_obj.digest()[:16], byteorder="big")

    def generate_sample(
        self,
        job_id: int,
        tag: str,
        problem_generator: Callable,
        statistics_calculator: Callable | None = None,
        batch_idx: int = 0,
    ) -> tuple[list[Any] | Any, list[Any] | Any, dict[str, Any] | None, float]:
        # Generate a unique seed for this job
        seed = self._generate_seed(job_id, tag, batch_idx)

        start_time = time()
        problem, solution = problem_generator(seed)
        runtime = time() - start_time

        if statistics_calculator is not None:
            sample_stats = statistics_calculator(problem, solution)
        else:
            sample_stats = None

        problem = (
            [self._convert_obj_to_str(p) for p in problem]
            if isinstance(problem, list)
            else self._convert_obj_to_str(problem)
        )
        solution = (
            [self._convert_obj_to_str(s) for s in solution]
            if isinstance(solution, list)
            else self._convert_obj_to_str(solution)
        )

        return problem, solution, sample_stats, runtime

    def _convert_poly_to_str(self, poly_str: str) -> str:
        """
        Convert sympy polynomial string representation to a more readable format.
        e.g., 2*x**2*y**2 -> 2*x^2*y^2
        5 mod 7*x**4*y**3 -> 5*x^4*y^3
        """
        # Remove mod (order) notation
        poly_str = re.sub(r" mod \d+", "", poly_str)
        # Replace ** with ^
        poly_str = re.sub(r"\*\*", "^", poly_str)
        return poly_str

    def _convert_obj_to_str(self, obj: Any) -> str:
        """
        Convert object to string, applying polynomial conversion if necessary.

        Args:
            obj: Object to convert

        Returns:
            String representation of the object
        """
        if isinstance(obj, PolyElement):
            return self._convert_poly_to_str(str(obj))
        return str(obj)

    def _run(
        self,
        tag: str,
        num_samples: int,
        problem_generator: Callable,
        statistics_calculator: Callable | None = None,
        dataset_writer: DatasetWriter | None = None,
        batch_size: int = 100000,
    ):
        """
        Generate a dataset with specified number of samples using parallel processing with batch writing.

        Args:
            tag: Dataset tag ("train" or "test")
            num_samples: Number of samples to generate
            problem_generator: Function to generate individual problems
            statistics_calculator: Optional function to calculate dataset statistics
            dataset_writer: DatasetWriter object for batch writing
            batch_size: Number of samples to process in each batch
        """
        start_time = time()

        # Initialize memory-efficient statistics calculator
        incremental_stats = MemoryEfficientStatisticsCalculator()

        # Validate batch size
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        # Calculate number of batches
        num_batches = (num_samples + batch_size - 1) // batch_size

        self.logger.info(
            f"---------------------------------- {tag} ----------------------------------"
        )
        self.logger.info(
            f"Dataset size: {num_samples} samples  (Batch size: {batch_size})\n"
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_samples)
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
                    delayed(self.generate_sample)(
                        batch_start + i,
                        tag,
                        problem_generator,
                        statistics_calculator,
                        batch_idx,
                    )
                    for i in range(current_batch_size)
                )
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx + 1}: {e}")
                raise

            # Unzip the results for current batch
            problems, solutions, sample_stats, runtimes = zip(*results)

            if self.verbose:
                self.logger.info("Parallel processing completed")

            # Store batch results for statistics only
            batch_samples = list(zip(problems, solutions))

            # Update statistics incrementally
            incremental_stats.update_batch(
                runtimes, sample_stats if sample_stats[0] is not None else []
            )

            # Write batch to file if dataset_writer is provided
            if dataset_writer is not None:
                dataset_writer.save_batch(
                    samples=batch_samples, tag=tag, batch_idx=batch_idx
                )

                if self.verbose:
                    self.logger.info(f"Batch {batch_idx + 1} saved to file")

            # Clear batch data from memory to prevent memory buildup
            del batch_samples, problems, solutions, sample_stats, runtimes

            if self.verbose:
                self.logger.info(f"Batch {batch_idx + 1}/{num_batches} completed")
                self.logger.info("")

        # Calculate overall statistics from incremental data
        total_time = time() - start_time

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
    ):
        """
        Generate datasets using parallel processing with batch writing.

        This method generates datasets based on the sizes dictionary.

        Examples:
            >>> # Single dataset
            >>> dataset_generator.run(
            ...     dataset_sizes={"train": 100000},
            ...     problem_generator=problem_generator,
            ...     statistics_calculator=statistics_calculator,
            ...     dataset_writer=dataset_writer,
            ...     batch_size=10000
            ... )
            >>> # Multiple datasets
            >>> dataset_generator.run(
            ...     dataset_sizes={"train": 100000, "test": 1000},
            ...     problem_generator=problem_generator,
            ...     statistics_calculator=statistics_calculator,
            ...     dataset_writer=dataset_writer,
            ...     batch_size=10000
            ... )

        Args:
            dataset_sizes: Dictionary mapping dataset names to number of samples.
                           Example: {"train": 100000, "test": 1000} or {"train": 100000}
            problem_generator: Function to generate individual problems
            statistics_calculator: Optional function to calculate dataset statistics
            dataset_writer: DatasetWriter object for batch writing
            batch_size: Number of samples to process in each batch
        """
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

        for dataset_name, num_samples in dataset_sizes.items():
            if dataset_name not in ["train", "test"]:
                raise ValueError(
                    f"Dataset name must be 'train' or 'test', got '{dataset_name}'"
                )
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
            self._run(tag=dataset_name, num_samples=num_samples, **common_args)

        self.logger.info("All datasets generated successfully!")
        self.logger.info(
            "==========================================================================\n"
        )
