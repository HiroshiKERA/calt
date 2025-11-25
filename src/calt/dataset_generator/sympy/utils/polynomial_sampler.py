import math
import random

import numpy as np
from sympy import GF, QQ, RR, ZZ
from sympy.core.mul import prod
from sympy.polys.domains.domain import Domain
from sympy.polys.orderings import MonomialOrder
from sympy.polys.rings import PolyElement, PolyRing, ring

from .single_polynomial_sampler import SinglePolynomialSampler


class PolynomialSampler:
    """Generator for random polynomials with specific constraints"""

    def __init__(
        self,
        symbols: str,
        field_str: str,
        order: str | MonomialOrder,
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
    ) -> None:
        """
        Initialize polynomial sampler

        Args:
            symbols: Symbols of polynomial ring
            field_str: Field of polynomial ring
            order: Order of polynomial ring
            max_num_terms: Maximum number of terms in polynomial. If None, all possible terms are allowed.
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
        self.single_poly_sampler = SinglePolynomialSampler()

    def get_field(self) -> Domain:
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

    def get_ring(self) -> PolyRing:
        """
        Generate polynomial ring

        Returns:
            PolyRing: Generated polynomial ring
        """

        R, *gens = ring(self.symbols, self.get_field(), self.order)
        return R

    def sample(
        self,
        num_samples: int = 1,
        size: tuple[int, int] | None = None,
        density: float = 1.0,
        matrix_type: str | None = None,
    ) -> list[PolyElement] | list[np.ndarray]:
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

    def _sample_polynomial(self) -> PolyElement:
        """Generate a single random polynomial"""
        # Determine degree
        if self.degree_sampling == "uniform":
            degree = random.randint(self.min_degree, self.max_degree)
        else:  # fixed
            degree = self.max_degree

        R = self.get_ring()

        # Determine number of terms
        max_possible_terms = math.comb(degree + R.ngens, degree)
        if self.max_num_terms is None:
            max_terms = max_possible_terms
        else:
            max_terms = min(self.max_num_terms, max_possible_terms)

        if self.term_sampling == "uniform":
            num_terms = random.randint(1, max_terms)
        else:  # fixed
            num_terms = max_terms

        # Generate polynomial with retry logic
        for attempt in range(self.max_attempts):
            p = self._generate_random_polynomial(degree, num_terms)

            # Check conditions
            if p == 0 and self.nonzero_instance:
                continue

            if self.total_degree(p) < self.min_degree:
                continue

            if not self.strictly_conditioned:
                break

            if self.total_degree(p) == degree and len(p.terms()) == num_terms:
                break

            if attempt == self.max_attempts - 1:
                raise RuntimeError(
                    f"Failed to generate polynomial satisfying conditions after {self.max_attempts} attempts"
                )

        return p

    def _generate_random_polynomial(self, degree: int, num_terms: int) -> PolyElement:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == "uniform"
        non_zero_coeff = self.nonzero_instance

        R = self.get_ring()
        field = R.domain

        if field == QQ:
            bound = self.num_bound if self.num_bound is not None else 10
            return self.single_poly_sampler.random_element(
                R=R,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
                num_bound=bound,
            )
        elif field in (RR, ZZ):
            coeff = self.max_coeff if self.max_coeff is not None else 10
            return self.single_poly_sampler.random_element(
                R=R,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
                min=-coeff,
                max=coeff,
            )
        elif field.is_FiniteField:  # Finite field
            return self.single_poly_sampler.random_element(
                R=R,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
            )

    def _sample_matrix(
        self,
        size: tuple[int, int],
        density: float = 1.0,
        matrix_type: str | None = None,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """Generate a matrix of random polynomials"""
        rows, cols = size
        num_entries = prod(size)

        # Generate polynomial entries
        entries = []
        for _ in range(num_entries):
            p = self._sample_polynomial(max_attempts)
            # Apply density
            if random.random() >= density:
                p = p * 0  # Use multiplication by 0 instead of R.zero
            entries.append(p)

        # Create matrix - use sympy Matrix with proper domain handling
        M = np.array(entries).reshape(rows, cols)

        # Apply special matrix type constraints
        if matrix_type == "unimodular_upper_triangular":
            for i in range(rows):
                for j in range(cols):
                    if i == j:
                        M[i, j] = 1
                    elif i > j:
                        M[i, j] = 0

        return M

    def total_degree(self, poly: PolyElement) -> int:
        """Compute total degree of a polynomial"""
        if poly.is_zero:
            return 0
        else:
            return max(sum(monom) for monom in poly.monoms())


def compute_max_coefficient(poly: PolyElement) -> float:
    """Compute maximum absolute coefficient value in a polynomial"""
    coeffs = poly.coeffs()
    field = poly.ring.domain

    if not coeffs:
        return 0

    if field == RR:
        return max(abs(float(c)) for c in coeffs)
    else:  # QQ case
        return max(
            max(abs(float(c.numerator)), abs(float(c.denominator))) for c in coeffs
        )


def compute_matrix_max_coefficient(M: np.ndarray) -> float:
    """Compute maximum absolute coefficient value in a polynomial matrix"""
    return max(compute_max_coefficient(p) for row in M for p in row)
