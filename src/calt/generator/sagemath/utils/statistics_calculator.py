from typing import Any, Dict, List, Union, Sequence, Literal
import numpy as np
from datetime import timedelta
from sage.all import PolynomialRing, QQ, RR


class StatisticsCalculator:
    """
    Calculate statistics for generated dataset.
    """

    def __init__(
        self,
        problem_type: Literal["polynomial", "numerical"],
        ring: PolynomialRing = None,
    ):
        """
        Initialize statistics calculator.

        Args:
            problem_type: Type of problems to generate ("polynomial" or "numerical")
            ring: Polynomial ring (required for polynomial problems)
        """
        if problem_type not in ["polynomial", "numerical"]:
            raise ValueError(
                f"Invalid problem type: {problem_type}. Must be either 'polynomial' or 'numerical'"
            )
        if problem_type == "polynomial" and ring is None:
            raise ValueError("Polynomial statistics require a polynomial ring")
        if problem_type == "numerical" and ring is not None:
            raise ValueError("Numerical statistics do not require a polynomial ring")

        self.problem_type = problem_type
        self.ring = ring

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
        if self.ring is None:
            raise ValueError("Polynomial statistics require a polynomial ring")

        # Initialize coefficients field and number of variables
        self.coeff_field = self.ring.base_ring()
        self.num_vars = self.ring.ngens()

        # Basic statistics
        num_polys = len(polys)

        if num_polys == 0:
            return {"num_polynomials": 0, "total_degree": 0, "total_terms": 0}

        # Calculate degrees
        degrees = [
            max(p.total_degree(), 0) for p in polys
        ]  # if polynomial p is zero, then p.total_degree() is -1, so we need to set it to 0

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
            "density": float(sum(num_terms))
            / (num_polys * (1 + max(degrees)) ** self.num_vars),
        }

        return stats

    def numerical_stats(self, numbers: List[Any]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of numbers.

        Args:
            numbers: List of numbers

        Returns:
            Dictionary containing statistical information about the numbers
        """
        # Convert to float for calculations
        values = [float(n) for n in numbers]

        stats = {
            "num_values": len(numbers),
            "max_value": max(values),
            "min_value": min(values),
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
            "sum_value": sum(values),
        }

        return stats

    def sample_stats(
        self,
        problem_input: Union[List[Any], Any],
        problem_output: Union[List[Any], Any],
        generation_time: timedelta,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a single generated sample.

        Args:
            problem_input: Input problem (polynomials/numbers or a single polynomial/number)
            problem_output: Output solution (polynomials/numbers or a single polynomial/number)
            generation_time: Time taken to generate this sample

        Returns:
            Dictionary containing statistics about the sample
        """
        if self.problem_type == "polynomial":
            # Polynomial statistics
            if isinstance(problem_input, list):
                input_stats = self.poly_stats(problem_input)
            else:
                input_stats = self.poly_stats([problem_input])
            if isinstance(problem_output, list):
                output_stats = self.poly_stats(problem_output)
            else:
                output_stats = self.poly_stats([problem_output])
        elif self.problem_type == "numerical":
            # Numerical statistics
            if isinstance(problem_input, list):
                input_stats = self.numerical_stats(problem_input)
            else:
                input_stats = self.numerical_stats([problem_input])
            if isinstance(problem_output, list):
                output_stats = self.numerical_stats(problem_output)
            else:
                output_stats = self.numerical_stats([problem_output])

        return {
            "generation_time": generation_time,
            "input": input_stats,
            "output": output_stats,
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

        # Aggregate statistics for input and output
        for key in ["input", "output"]:
            overall_stats = {}
            for stat_key in sample_stats[0][key].keys():
                values = [s[key][stat_key] for s in sample_stats]
                overall_stats[stat_key] = self._calculate_statistics(values)
            stats[f"{key}_overall"] = overall_stats

        return stats
