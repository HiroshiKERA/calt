from abc import ABC, abstractmethod
from typing import Any


class BaseStatisticsCalculator(ABC):
    """
    Abstract base class for sample-level statistics calculators.

    This class defines the interface for calculators that compute
    statistics for individual samples (problems and solutions).
    """

    @abstractmethod
    def __call__(
        self, problem: Any, solution: Any
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for a single sample.

        Args:
            problem: The problem data
            solution: The solution data

        Returns:
            Dictionary with keys "problem" and "solution", each mapping to a sub-dictionary
            containing descriptive statistics. The structure should be:
            {"problem": {"metric1": value1, "metric2": value2, ...},
             "solution": {"metric1": value1, "metric2": value2, ...}}

            Example:
            {"problem": {"total_degree": 2, "num_polynomials": 3},
             "solution": {"total_degree": 3, "num_polynomials": 3}}
        """
        pass


class IncrementalStatistics:
    """
    Calculate statistics incrementally without storing all data in memory.

    This class implements Welford's online algorithm for calculating mean and variance
    without storing all data points. The standard deviation is calculated as the
    population standard deviation (sqrt(variance)).

    Reference: Welford, B. P. (1962). "Note on a method for calculating corrected sums
    of squares and products". Technometrics. 4 (3): 419-420.
    """

    def __init__(self):
        """Initialize incremental statistics calculator."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean (Welford's algorithm)
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, value: float) -> None:
        """
        Update statistics with a new value using Welford's online algorithm.

        This method implements the core of Welford's algorithm:
        1. Update count
        2. Calculate delta from old mean
        3. Update mean incrementally
        4. Update M2 (sum of squared differences) for variance calculation

        Args:
            value: New value to include in statistics
        """
        self.count += 1
        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2  # Accumulate squared differences for variance
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def update_batch(self, values: list[float]) -> None:
        """
        Update statistics with a batch of values.

        Args:
            values: List of values to include in statistics
        """
        for value in values:
            self.update(value)

    def get_statistics(self) -> dict[str, float]:
        """
        Get current statistics calculated using Welford's algorithm.

        The variance is calculated as M2 / count, where M2 is the accumulated
        sum of squared differences from the mean (Welford's algorithm).
        The standard deviation (std) is the population standard deviation,
        calculated as sqrt(variance).

        Returns:
            Dictionary containing:
            - mean: Arithmetic mean of all values
            - std: Population standard deviation (sqrt(variance))
            - min: Minimum value observed
            - max: Maximum value observed
        """
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        # Calculate variance using Welford's accumulated M2 value
        variance = self.M2 / self.count if self.count > 1 else 0.0
        std = variance**0.5 if variance >= 0 else 0.0

        return {
            "mean": self.mean,
            "std": std,
            "min": self.min_val if self.min_val != float("inf") else 0.0,
            "max": self.max_val if self.max_val != float("-inf") else 0.0,
        }


class MemoryEfficientStatisticsCalculator:
    """
    Memory-efficient statistics calculator that uses incremental computation.

    This calculator avoids storing all data in memory by computing statistics
    incrementally as batches are processed using Welford's online algorithm
    for numerical stability and memory efficiency. All standard deviations
    are calculated as population standard deviations.
    """

    def __init__(self):
        """Initialize incremental sample statistics calculator."""
        self.runtime_stats = IncrementalStatistics()
        self.sample_stats = {}  # Store aggregated sample statistics by category

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
