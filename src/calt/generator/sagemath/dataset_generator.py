from typing import Any, Dict, List, Tuple, Callable, Union, Literal
from joblib import Parallel, delayed
from time import time
import hashlib
from sage.all import PolynomialRing
from .utils.statistics_calculator import (
    StatisticsCalculator,
)


class DatasetGenerator:
    """Base class for problem generators"""

    def __init__(
        self,
        problem_type: Literal["polynomial", "numerical"],
        ring: PolynomialRing = None,
        n_jobs: int = -1,
        verbose: bool = True,
        root_seed: int = 42,
    ):
        """
        Initialize problem generator.

        Args:
            problem_type: Type of problems to generate ("polynomial" or "numerical")
            ring: Polynomial ring (required for polynomial problems)
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to display progress information
            root_seed: Root seed for reproducibility
        """
        if problem_type not in ["polynomial", "numerical"]:
            raise ValueError("Invalid problem type")
        if problem_type == "polynomial" and ring is None:
            raise ValueError("Polynomial problems require a polynomial ring")
        if problem_type == "numerical" and ring is not None:
            raise ValueError("Numerical problems do not require a polynomial ring")

        self.problem_type = problem_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.root_seed = root_seed
        # Initialize statistics calculator
        self.stats_calculator = StatisticsCalculator(problem_type, ring)

    def _generate_seed(self, job_id: int, train: bool) -> int:
        """
        Generate a unique seed value for each job using SHA-256 hash.
        Uses 8 bytes (64 bits) of the hash to ensure extremely low collision probability.

        Args:
            job_id: Job identifier
            train: Whether this is for training data

        Returns:
            Integer seed value (64 bits)
        """
        # Create a unique string for this job
        seed_str = f"{self.root_seed}_{'train' if train else 'test'}_{job_id}"
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(seed_str.encode())
        # Convert first 8 bytes to integer (64-bit)
        return int.from_bytes(hash_obj.digest()[:8], byteorder="big")

    def generate_sample(
        self, problem_generator: Callable, job_id: int, train: bool
    ) -> Tuple[Union[List[Any], Any], Union[List[Any], Any], Dict[str, Any]]:
        start_time = time()
        # Generate a unique seed for this job
        seed = self._generate_seed(job_id, train)
        problem_input, problem_output = problem_generator(seed)
        sample_stats = self.stats_calculator.sample_stats(
            problem_input, problem_output, time() - start_time
        )

        return problem_input, problem_output, sample_stats

    def run(
        self, num_samples: int, problem_generator: Callable, train: bool
    ) -> Tuple[
        List[Tuple[Union[List[Any], Any], Union[List[Any], Any]]], Dict[str, Any]
    ]:
        """
        Generate multiple samples using parallel processing.

        Args:
            num_samples: Number of samples to generate
            problem_generator: Function to generate individual problems
            train: Whether this is for training data

        Returns:
            Tuple containing (list of samples, overall statistics)
        """
        start_time = time()

        # Generate samples in parallel using joblib
        results = Parallel(
            n_jobs=self.n_jobs, backend="multiprocessing", verbose=self.verbose
        )(
            delayed(self.generate_sample)(problem_generator, i, train)
            for i in range(num_samples)
        )

        # Unzip the results
        problem_inputs, problem_outputs, sample_stats = zip(*results)

        # Calculate overall statistics
        total_time = time() - start_time
        overall_stats = self.stats_calculator.overall_stats(
            sample_stats, total_time=total_time, num_samples=num_samples
        )

        return list(zip(problem_inputs, problem_outputs)), overall_stats
