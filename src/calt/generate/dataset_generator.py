from typing import Any, Dict, List, Tuple, Callable, Union, Optional
from joblib import Parallel, delayed
from time import time
import hashlib
from datetime import timedelta
import numpy as np


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
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to display progress information
            root_seed: Root seed for reproducibility
        """

        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.root_seed = root_seed

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
        self,
        job_id: int,
        train: bool,
        problem_generator: Callable,
        statistics_calculator: Optional[Callable] = None,
    ) -> Tuple[Union[List[Any], Any], Union[List[Any], Any], Dict[str, Any], timedelta]:
        # Generate a unique seed for this job
        seed = self._generate_seed(job_id, train)

        start_time = time()
        problem_input, problem_output = problem_generator(seed)
        runtime = time() - start_time

        if statistics_calculator is not None:
            sample_stats = statistics_calculator(problem_input, problem_output)
        else:
            sample_stats = None

        return problem_input, problem_output, sample_stats, runtime

    def run(
        self,
        train: bool,
        num_samples: int,
        problem_generator: Callable,
        statistics_calculator: Optional[Callable] = None,
    ) -> Tuple[
        List[Tuple[Union[List[Any], Any], Union[List[Any], Any]]], Dict[str, Any]
    ]:
        """
        Generate multiple samples using parallel processing.

        Args:
            num_samples: Number of samples to generate
            train: Whether this is for training data
            problem_generator: Function to generate individual problems
            statistics_calculator: Optional function to calculate dataset statistics

        Returns:
            Tuple containing (list of samples, overall statistics)
        """
        start_time = time()

        # Generate samples in parallel using joblib
        results = Parallel(
            n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
        )(
            delayed(self.generate_sample)(
                i, train, problem_generator, statistics_calculator
            )
            for i in range(num_samples)
        )

        # Unzip the results
        problem_inputs, problem_outputs, sample_stats, runtimes = zip(*results)

        # Calculate overall statistics
        total_time = time() - start_time
        if statistics_calculator is not None:
            overall_stats = statistics_calculator.overall_stats(
                sample_stats=sample_stats,
                runtimes=runtimes,
                total_time=total_time,
                num_samples=num_samples,
            )
        else:
            overall_stats = {
                "total_time": total_time,
                "num_samples": num_samples,
                "samples_per_second": num_samples / total_time,
                "generation_time": {
                    "mean": float(sum(runtimes) / len(runtimes)),
                    "std": float(np.std(runtimes)),
                    "min": float(np.min(runtimes)),
                    "max": float(np.max(runtimes)),
                },
            }

        return list(zip(problem_inputs, problem_outputs)), overall_stats
