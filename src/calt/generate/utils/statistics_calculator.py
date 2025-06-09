from typing import Any, Dict, List, Sequence, Protocol
import numpy as np
from datetime import timedelta


class BaseStatisticsCalculator(Protocol):
    """Base class for statistics calculators."""

    def _calculate_statistics(self, values: Sequence[float]) -> Dict[str, float]:
        """
        Calculate basic statistics (mean, std, min, max) for a sequence of values.
        """
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def overall_stats(
        self,
        sample_stats: List[Dict[str, Any]],
        runtimes: List[timedelta],
        total_time: timedelta,
        num_samples: int,
    ) -> Dict[str, Any]:
        """Calculate overall statistics from all generated samples."""
        stats = {
            "total_time": total_time,
            "samples_per_second": num_samples / total_time,
            "num_samples": num_samples,
        }

        # Aggregate statistics for generation time
        stats["generation_time"] = self._calculate_statistics(runtimes)

        # Aggregate statistics for input and output
        for key in ["input", "output"]:
            overall_stats = {}
            for stat_key in sample_stats[0][key].keys():
                values = [s[key][stat_key] for s in sample_stats]
                overall_stats[stat_key] = self._calculate_statistics(values)
            stats[f"{key}_overall"] = overall_stats

        return stats
