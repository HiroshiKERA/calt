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
    """Generator for random polynomials with specific constraints (SymPy).

    The sampler builds polynomials by choosing a target degree and number
    of terms (within min/max bounds), then uses :class:`SinglePolynomialSampler`
    to select that many distinct monomials and assign random coefficients
    from the base ring. Ring is specified by symbols, field_str, and order.

    Behavior summary
    -----------------

    **degree_sampling** controls how monomial degrees are chosen (passed
    as ``choose_degree`` to the internal sampler):

    - ``'uniform'``: For each term, a degree in [min_degree, max_degree]
      is chosen uniformly at random, then a monomial of that degree is
      chosen. The resulting polynomial's degree distribution is more
      uniform over the range.
    - ``'fixed'``: Monomials are chosen uniformly from all monomials of
      degree at most max_degree. The polynomial tends to have total degree
      equal to max_degree.

    **Degree and number of terms**: Every returned polynomial has total
    degree >= min_degree. The guarantees on total degree and number of
    terms depend on ``strictly_conditioned`` and ``nonzero_instance``;
    see the constructor parameters for details.
    """

    def __init__(
        self,
        symbols: str,
        field_str: str,
        order: str | MonomialOrder = "grevlex",
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
        Initialize polynomial sampler.

        Args:
            symbols: Variable names for the polynomial ring.
            field_str: Base ring specifier: "QQ", "RR", "ZZ", or "GF(p)"
                for a prime finite field.
            order: Term order of the ring, e.g. "grevlex".
            max_num_terms: Upper bound on number of terms. If None, all
                monomials of the chosen degree are allowed.
            max_degree: Maximum total degree of the polynomial.
            min_degree: Minimum total degree; every returned polynomial
                has total degree >= min_degree.
            max_coeff: Bound on coefficient absolute value for RR and ZZ.
            num_bound: Bound on numerator/denominator absolute value
                for QQ.
            degree_sampling: ``'uniform'`` or ``'fixed'``; see class
                docstring (Behavior summary).
            term_sampling: ``'uniform'``: number of terms chosen uniformly
                in [1, max_terms] (max_terms bounded by max_num_terms);
                ``'fixed'``: use max_terms.
            strictly_conditioned: Controls when a generated polynomial
                is accepted.

                - If True:
                    - Return only when total degree equals the degree
                      selected for this sample and number of terms
                      equals the number of terms selected for this
                      sample. (Those values are chosen by
                      degree_sampling and term_sampling; degree is in
                      [min_degree, max_degree], and number of terms is
                      at most max_num_terms.)
                    - RuntimeError is raised if no success within
                      max_attempts.
                - If False:
                    - Return the first polynomial with total degree >=
                      min_degree and (if nonzero_instance) non-zero.
                    - Number of terms may be less than the chosen value
                      when nonzero_instance is False.
            nonzero_instance: If True, the zero polynomial is never
                returned and all coefficients are non-zero (predictable
                number of terms). If False, coefficients may be zero.
            max_attempts: Maximum trials per polynomial when
                strictly_conditioned is True; RuntimeError is raised if
                no success.
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
        """Return the SymPy domain for field_str (QQ, RR, ZZ, or GF(p))."""
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
        """Return the polynomial ring (PolyRing) for the configured symbols, field, and order."""

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
            List of polynomials, or list of polynomial matrices when
            size is provided.
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
        """Generate a random polynomial with the given degree and number of terms via SinglePolynomialSampler.random_element."""
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
    ) -> np.ndarray:
        """
        Generate a matrix of random polynomials.

        Args:
            size: (rows, cols) shape of the matrix.
            density: Probability that each entry is non-zero (entries
                may be zeroed at random).
            matrix_type: If "unimodular_upper_triangular", set diagonal
                to 1 and strict lower part to 0.

        Returns:
            An array of shape size with polynomial entries from this
            sampler.
        """
        rows, cols = size
        num_entries = prod(size)

        # Generate polynomial entries
        entries = []
        for _ in range(num_entries):
            p = self._sample_polynomial()
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
        """Return the total degree of the polynomial."""
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
