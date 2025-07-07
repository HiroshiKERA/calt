from typing import Any
import random
from sage.all import (
    PolynomialRing,
    QQ,
    RR,
    ZZ,
    GF,
    matrix,
    binomial,
    randint,
    prod,
    TermOrder,
)


class PolynomialSampler:
    """Generator for random polynomials with specific constraints"""

    def __init__(
        self,
        symbols: str,
        field_str: str,
        order: str | TermOrder,
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
    ):
        """
        Initialize polynomial sampler

        Args:
            symbols: Symbols of polynomial ring
            field_str: Field of polynomial ring
            order: Order of polynomial ring
            max_num_terms: Maximum number of terms in polynomial
            max_degree: Maximum degree of polynomial
            min_degree: Minimum degree of polynomial
            max_coeff: Maximum coefficient value
            num_bound: Maximum absolute value of coefficients
            degree_sampling: How to sample degree ('uniform' or 'fixed')
            term_sampling: How to sample number of terms ('uniform' or 'fixed')
            strictly_conditioned: Whether to strictly enforce conditions
            nonzero_instance: Whether to enforce non-zero instance
            max_attempts: Maximum number of attempts to generate a polynomial satisfying conditions
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

    def get_field(self):
        """Convert field_str to actual sympy domain object"""
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

    def get_ring(self) -> PolynomialRing:
        """
        Generate polynomial ring

        Returns:
            PolynomialRing: Generated polynomial ring
        """
        R = PolynomialRing(self.get_field(), self.symbols, order=self.order)
        return R

    def sample(
        self,
        num_samples: int = 1,
        size: tuple[int, int] | None = None,
        density: float = 1.0,
        matrix_type: str | None = None,
    ) -> list[Any] | list[matrix]:
        """
        Generate random polynomial samples

        Args:
            num_samples: Number of samples to generate
            size: If provided, generate matrix of polynomials with given size
            density: Probability of non-zero entries in matrix
            matrix_type: Special matrix type (e.g., 'unimodular_upper_triangular')

        Returns:
            List of polynomials or polynomial matrices
        """
        if size is not None:
            return [
                self._sample_matrix(size, density, matrix_type)
                for _ in range(num_samples)
            ]
        else:
            return [self._sample_polynomial() for _ in range(num_samples)]

    def _sample_polynomial(self) -> Any:
        """Generate a single random polynomial"""
        # Determine degree
        if self.degree_sampling == "uniform":
            degree = randint(self.min_degree, self.max_degree)
        else:  # fixed
            degree = self.max_degree

        R = self.get_ring()

        # Determine number of terms
        max_possible_terms = binomial(degree + R.ngens(), degree)
        if self.max_num_terms is None:
            max_terms = max_possible_terms
        else:
            max_terms = min(self.max_num_terms, max_possible_terms)

        if self.term_sampling == "uniform":
            num_terms = randint(1, max_terms)
        else:  # fixed
            num_terms = max_terms

        # Generate polynomial with retry logic
        for attempt in range(self.max_attempts):
            p = self._generate_random_polynomial(degree, num_terms)

            # Check conditions
            if p == 0 and self.nonzero_instance:
                continue

            if p.total_degree() < self.min_degree:
                continue

            if not self.strictly_conditioned:
                break

            if p.total_degree() == degree and len(p.monomials()) == num_terms:
                break

            if attempt == self.max_attempts - 1:
                raise RuntimeError(
                    f"Failed to generate polynomial satisfying conditions after {self.max_attempts} attempts"
                )

        return p

    def _generate_random_polynomial(self, degree: int, num_terms: int) -> Any:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == "uniform"

        R = self.get_ring()
        field = R.base_ring()

        if field == QQ:
            bound = self.num_bound if self.num_bound is not None else 10
            return R.random_element(
                degree=degree,
                terms=num_terms,
                num_bound=bound,
                choose_degree=choose_degree,
            )
        elif field == RR:
            coeff = self.max_coeff if self.max_coeff is not None else 10
            return R.random_element(
                degree=degree,
                terms=num_terms,
                min=-coeff,
                max=coeff,
                choose_degree=choose_degree,
            )
        elif field == ZZ:
            coeff = self.max_coeff if self.max_coeff is not None else 10
            return R.random_element(
                degree=degree,
                terms=num_terms,
                x=-coeff,
                y=coeff + 1,
                choose_degree=choose_degree,
            )
        else:  # Finite field
            return R.random_element(
                degree=degree, terms=num_terms, choose_degree=choose_degree
            )

    def _sample_matrix(
        self,
        size: tuple[int, int],
        density: float = 1.0,
        matrix_type: str | None = None,
        max_attempts: int = 100,
    ) -> matrix:
        """Generate a matrix of random polynomials"""
        rows, cols = size
        num_entries = prod(size)
        R = self.get_ring()

        # Generate polynomial entries
        entries = []
        for _ in range(num_entries):
            p = self._sample_polynomial(max_attempts)
            # Apply density
            if random.random() >= density:
                p *= 0
            entries.append(p)

        # Create matrix
        M = matrix(R, rows, cols, entries)

        # Apply special matrix type constraints
        if matrix_type == "unimodular_upper_triangular":
            for i in range(rows):
                for j in range(cols):
                    if i == j:
                        M[i, j] = 1
                    elif i > j:
                        M[i, j] = 0

        return M


def compute_max_coefficient(poly: Any) -> int:
    """Compute maximum absolute coefficient value in a polynomial"""
    coeffs = poly.coefficients()
    field = poly.base_ring()

    if not coeffs:
        return 0

    if field == RR:
        return max(abs(c) for c in coeffs)
    else:  # QQ case
        return max(max(abs(c.numerator()), abs(c.denominator())) for c in coeffs)


def compute_matrix_max_coefficient(M: matrix) -> int:
    """Compute maximum absolute coefficient value in a polynomial matrix"""
    return max(compute_max_coefficient(p) for p in M.list())
