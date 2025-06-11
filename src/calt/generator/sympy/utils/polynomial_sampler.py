import math
import random
from typing import Optional, List, Union, Tuple
from sympy import ZZ, QQ, RR
from sympy.polys.rings import PolyRing, PolyElement
from .single_polynomial_sampler import SinglePolynomialSampler


class PolynomialSampler:
    """Generator for random polynomials with specific constraints"""

    def __init__(
        self,
        ring: PolyRing,
        max_num_terms: int = 10,
        max_degree: int = 5,
        min_degree: int = 0,
        degree_sampling: str = "uniform",  # 'uniform' or 'fixed'
        term_sampling: str = "uniform",  # 'uniform' or 'fixed'
        max_coeff: Optional[int] = None,  # Used for RR and ZZ
        num_bound: Optional[int] = None,  # Used for QQ
        strictly_conditioned: bool = True,
        nonzero_instance: bool = False,
    ):
        """
        Initialize polynomial sampler

        Args:
            ring: SymPy polynomial ring
            max_num_terms: Maximum number of terms in polynomial
            max_degree: Maximum degree of polynomial
            min_degree: Minimum degree of polynomial
            max_coeff: Maximum coefficient value
            num_bound: Maximum absolute value of coefficients
            degree_sampling: How to sample degree ('uniform' or 'fixed')
            term_sampling: How to sample number of terms ('uniform' or 'fixed')
            strictly_conditioned: Whether to strictly enforce conditions
            nonzero_instance: Whether to enforce non-zero instance
        """
        self.ring = ring
        self.field = ring.domain
        self.max_num_terms = max_num_terms
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.degree_sampling = degree_sampling
        self.term_sampling = term_sampling
        self.strictly_conditioned = strictly_conditioned
        self.nonzero_instance = nonzero_instance

        # Set coefficients based on field type
        self.max_coeff = max_coeff if self.field in (RR, ZZ) else None
        self.num_bound = num_bound if self.field == QQ else None

        # Set default values if not provided
        if self.max_coeff is None and self.field in (RR, ZZ):
            self.max_coeff = 10

        if self.num_bound is None and self.field == QQ:
            self.num_bound = 10

        self.single_poly_sampler = SinglePolynomialSampler()

    def sample(
        self,
        num_samples: int = 1,
        size: Optional[Tuple[int, int]] = None,
        density: float = 1.0,
        matrix_type: Optional[str] = None,
    ) -> Union[List[PolyElement], List[List[List[PolyElement]]]]:
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

    def _sample_polynomial(self, max_attempts: int = 100) -> PolyElement:
        """Generate a single random polynomial"""
        # Determine degree
        if self.degree_sampling == "uniform":
            degree = random.randint(self.min_degree, self.max_degree)
        else:  # fixed
            degree = self.max_degree

        # Determine number of terms
        max_possible_terms = math.comb(degree + self.ring.ngens, degree)
        max_terms = min(self.max_num_terms, max_possible_terms)

        if self.term_sampling == "uniform":
            num_terms = random.randint(1, max_terms)
        else:  # fixed
            num_terms = max_terms

        # Generate polynomial with retry logic
        for attempt in range(max_attempts):
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

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to generate polynomial satisfying conditions after {max_attempts} attempts"
                )

        return p

    def _generate_random_polynomial(self, degree: int, num_terms: int) -> PolyElement:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == "uniform"
        non_zero_coeff = self.nonzero_instance

        if self.field == QQ:
            return self.single_poly_sampler.random_element(
                self.ring,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
                num_bound=self.num_bound,
            )
        elif self.field in (RR, ZZ):
            return self.single_poly_sampler.random_element(
                self.ring,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
                min=-self.max_coeff,
                max=self.max_coeff,
            )
        else:  # Finite field
            return self.single_poly_sampler.random_element(
                self.ring,
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree,
                non_zero_coeff=non_zero_coeff,
            )

    def _sample_matrix(
        self,
        size: Tuple[int, int],
        density: float = 1.0,
        matrix_type: Optional[str] = None,
        max_attempts: int = 100,
    ) -> List[List[PolyElement]]:
        """Generate a matrix of random polynomials"""
        rows, cols = size
        matrix = [[None for _ in range(cols)] for _ in range(rows)]

        # Generate polynomial entries
        for i in range(rows):
            for j in range(cols):
                if random.random() < density:
                    p = self._sample_polynomial(max_attempts)
                    matrix[i][j] = p
                else:
                    matrix[i][j] = self.ring.zero

        # Apply special matrix type constraints
        if matrix_type == "unimodular_upper_triangular":
            for i in range(rows):
                for j in range(cols):
                    if i == j:
                        matrix[i][j] = self.ring.one
                    elif i > j:
                        matrix[i][j] = self.ring.zero

        return matrix

    def total_degree(self, poly: PolyElement) -> int:
        """Compute total degree of a polynomial"""
        if poly.is_zero:
            return 0
        else:
            return max(list(sum(monom) for monom in poly.monoms()))


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


def compute_matrix_max_coefficient(M: List[List[PolyElement]]) -> float:
    """Compute maximum absolute coefficient value in a polynomial matrix"""
    return max(compute_max_coefficient(p) for row in M for p in row)
