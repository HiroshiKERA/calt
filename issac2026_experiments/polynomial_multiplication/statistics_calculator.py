from sage.all import QQ, RR, ZZ
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular

from calt.dataset.utils.statistics_calculator import BaseStatisticsCalculator


class PolyStatisticsCalculator(BaseStatisticsCalculator):
    """
    Statistics calculator for polynomial problems.
    """

    def __call__(
        self,
        problem: list[MPolynomial_libsingular] | MPolynomial_libsingular,
        answer: list[MPolynomial_libsingular] | MPolynomial_libsingular,
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for a single generated sample.

        Args:
            problem: Either a list of polynomials or a single polynomial
            answer: Either a list of polynomials or a single polynomial

        Returns:
            Dictionary with keys "problem" and "answer", each mapping to a sub-dictionary
            containing descriptive statistics including:
            - num_polynomials: Number of polynomials in the system
            - sum_total_degree: Sum of total degrees of all polynomials in the system
            - min_total_degree: Minimum degree of any polynomial in the system
            - max_total_degree: Maximum degree of any polynomial in the system
            - sum_num_terms: Total number of terms across all polynomials in the system
            - min_num_terms: Minimum number of terms in any polynomial in the system
            - max_num_terms: Maximum number of terms in any polynomial in the system
            - min_abs_coeff: Minimum absolute coefficient value in the system
            - max_abs_coeff: Maximum absolute coefficient value in the system

        Examples:
            >>> stats_calculator = PolyStatisticsCalculator()
            >>> stats = stats_calculator(problem=[x^2 + 1, x^3 + 2], answer=[x^2 + 1, x^3 + 2])
            >>> stats['problem']['num_polynomials']
            2
            >>> stats['answer']['num_polynomials']
            2
        """
        return {
            "problem": self.poly_system_stats(
                problem if isinstance(problem, list) else [problem]
            ),
            "answer": self.poly_system_stats(
                answer if isinstance(answer, list) else [answer]
            ),
        }

    def _extract_coefficients(self, poly: MPolynomial_libsingular) -> list[float | int]:
        """Extract coefficients from polynomial based on field type."""
        coeff_field = poly.parent().base_ring()
        if coeff_field == QQ:
            return [abs(float(c.numerator())) for c in poly.coefficients()] + [
                abs(float(c.denominator())) for c in poly.coefficients()
            ]
        elif coeff_field in (RR, ZZ):
            return [abs(float(c)) for c in poly.coefficients()]
        elif coeff_field.is_field() and coeff_field.characteristic() > 0:
            return [int(c) for c in poly.coefficients()]
        return []

    def poly_system_stats(
        self, polys: list[MPolynomial_libsingular]
    ) -> dict[str, int | float]:
        """
        Calculate statistics for a list of polynomials.

        Args:
            polys: List of polynomials

        Returns:
            Dictionary containing statistical information about the polynomials
        """
        if not polys:
            raise ValueError(
                "Cannot calculate statistics for empty list of polynomials"
            )

        # Clamp to 0 since zero polynomial has degree()/total_degree() == -1
        def _degree(p):
            if p.parent().ngens() == 1:
                return int(max(p.degree(), 0))  # univariate
            return int(max(p.total_degree(), 0))  # multivariate

        degrees = [_degree(p) for p in polys]
        num_terms = [len(p.monomials()) for p in polys]
        coeffs = [c for p in polys for c in self._extract_coefficients(p)]

        return {
            # System size statistics
            "num_polynomials": len(polys),
            # Degree statistics
            "sum_total_degree": sum(degrees),
            "min_total_degree": min(degrees),
            "max_total_degree": max(degrees),
            # Term count statistics
            "sum_num_terms": sum(num_terms),
            "min_num_terms": min(num_terms),
            "max_num_terms": max(num_terms),
            # Coefficient statistics
            "min_abs_coeff": min(coeffs) if coeffs else 0,
            "max_abs_coeff": max(coeffs) if coeffs else 0,
        }
