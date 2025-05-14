from typing import Any, Dict, List, Union, Sequence
import numpy as np
from datetime import timedelta
from sage.all import PolynomialRing, QQ, RR


class StatisticsCalculator:
    """
    Calculate statistics for generated dataset.
    """

    def __init__(self, ring: PolynomialRing):
        self.coeff_field = ring.base_ring()
        self.num_vars = ring.ngens()

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

    def poly_stats(self, polys: List[Any]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of polynomials.

        Args:
            polys: List of polynomials

        Returns:
            Dictionary containing statistical information about the polynomial system
        """
        # Basic statistics
        num_polys = len(polys)

        if num_polys == 0:
            return {"num_polynomials": 0, "total_degree": 0, "total_terms": 0}

        # Calculate degrees
        degrees = [p.total_degree() for p in polys]

        # Calculate number of terms
        num_terms = [len(p.monomials()) for p in polys]

        # Calculate coefficient statistics
        coeffs = []
        for p in polys:
            if self.coeff_field == QQ:
                # For QQ, consider both numerators(分子) and denominators(分母)
                coeffs.extend([abs(c.numerator()) for c in p.coefficients()])
                coeffs.extend([abs(c.denominator()) for c in p.coefficients()])
            elif self.coeff_field == RR:
                # For RR, take absolute values
                coeffs.extend([abs(c) for c in p.coefficients()])
            else:  # GF
                # For finite fields, just take the values
                coeffs.extend([int(c) for c in p.coefficients()])

        stats = {
            # System size statistics
            "num_polynomials": num_polys,
            "total_degree": sum(degrees),
            "total_terms": sum(num_terms),
            # Degree statistics
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            # "avg_degree": float(np.mean(degrees)),
            # "std_degree": float(np.std(degrees)),
            # Term count statistics
            "max_terms": max(num_terms),
            "min_terms": min(num_terms),
            # "avg_terms": float(np.mean(num_terms)),
            # "std_terms": float(np.std(num_terms)),
            # Coefficient statistics
            "max_coeff": max(coeffs) if coeffs else 0,
            "min_coeff": min(coeffs) if coeffs else 0,
            # "avg_coeff": float(np.mean(coeffs)) if coeffs else 0,
            # "std_coeff": float(np.std(coeffs)) if coeffs else 0,
            # Additional system properties
            # "density": float(sum(num_terms))
            # / (num_polys * (1 + max(degrees)) ** self.num_vars),
        }

        return stats

    def sample_stats(
        self,
        F: Union[List[Any], Any],
        G: Union[List[Any], Any],
        generation_time: timedelta,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a single generated sample.

        Args:
            F: List of input polynomials or a single input polynomial
            G: List of output polynomials or a single output polynomial
            generation_time: Time taken to generate this sample

        Returns:
            Dictionary containing statistics about the sample
        """
        if isinstance(F, list):
            F_stats = self.poly_stats(F)
        else:
            F_stats = self.poly_stats([F])
        if isinstance(G, list):
            G_stats = self.poly_stats(G)
        else:
            G_stats = self.poly_stats([G])

        return {
            "generation_time": generation_time,
            "input_polynomials": F_stats,
            "output_polynomials": G_stats,
        }

    def overall_stats(
        self,
        sample_stats: List[Dict[str, Any]],
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
        values = [s["generation_time"] for s in sample_stats]
        stats["generation_time"] = self._calculate_statistics(values)

        # Aggregate statistics for input polynomials
        input_stats = {}
        for key in sample_stats[0]["input_polynomials"].keys():
            values = [s["input_polynomials"][key] for s in sample_stats]

            input_stats[f"{key}"] = self._calculate_statistics(values)

        # Aggregate statistics for output polynomials
        output_stats = {}
        for key in sample_stats[0]["output_polynomials"].keys():
            values = [s["output_polynomials"][key] for s in sample_stats]

            output_stats[f"{key}"] = self._calculate_statistics(values)

        stats["input_polynomials_overall"] = input_stats
        stats["output_polynomials_overall"] = output_stats

        return stats
