from typing import Any

from sage.all import (
    GF,
    QQ,
    RR,
    ZZ,
    PolynomialRing,
    TermOrder,
    binomial,
    matrix,
    prod,
    randint,
)
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular


class PolynomialSampler:
    """Generator for random polynomials with specific constraints"""

    def __init__(
        self,
        symbols: str | None = None,
        field_str: str | None = None,
        order: str | TermOrder | None = None,
        ring: Any = None,
        max_num_terms: int | None = 10,
        max_degree: int = 5,
        min_degree: int = 0,
        degree_sampling: str = "uniform",  # 'uniform' or 'fixed'
        term_sampling: str = "uniform",  # 'uniform' or 'fixed'
        max_coeff: int | None = None,  # Used for RR and ZZ
        num_bound: int | None = None,  # Used for QQ
        strictly_conditioned: bool = True,
        nonzero_instance: bool = True,
        nonzero_coeff: bool = False,  # Whether to exclude zero coefficients
        max_attempts: int = 1000,
    ):
        """
        Initialize polynomial sampler

        Args:
            symbols: Symbols of polynomial ring (required if ring is None)
            field_str: Field of polynomial ring (required if ring is None)
            order: Order of polynomial ring (required if ring is None)
            ring: PolynomialRing object (alternative to symbols/field_str/order)
            max_num_terms: Maximum number of terms in polynomial. If None, all possible terms are allowed.
            max_degree: Maximum degree of polynomial
            min_degree: Minimum degree of polynomial
            max_coeff: Maximum coefficient value (used for RR and ZZ)
            num_bound: Maximum absolute value of coefficients (used for QQ)
            degree_sampling: How to sample degree ('uniform' or 'fixed')
            term_sampling: How to sample number of terms ('uniform' or 'fixed')
            strictly_conditioned: Whether to strictly enforce conditions
            nonzero_instance: Whether to enforce non-zero instance
            nonzero_coeff: Whether to exclude zero coefficients during coefficient generation
            max_attempts: Maximum number of attempts to generate a polynomial satisfying conditions
        """
        # Validate input parameters
        if ring is not None:
            if symbols is not None or field_str is not None or order is not None:
                raise ValueError("Cannot specify both ring and symbols/field_str/order")
            self.ring = ring
            self.symbols = None
            self.field_str = None
            self.order = None
        else:
            if symbols is None or field_str is None or order is None:
                raise ValueError(
                    "Must specify either ring or all of symbols/field_str/order"
                )
            self.ring = None
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
        self.nonzero_coeff = nonzero_coeff
        self.max_attempts = max_attempts

    def get_field(self):
        """Convert field_str to actual sympy domain object"""
        if self.ring is not None:
            return self.ring.base_ring()

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
        if self.ring is not None:
            return self.ring

        R = PolynomialRing(self.get_field(), self.symbols, order=self.order)
        return R

    def sample(
        self,
        num_samples: int = 1,
        size: tuple[int, int] | None = None,
        density: float = 1.0,
        matrix_type: str | None = None,
    ) -> list[MPolynomial_libsingular] | list[matrix]:
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

    def _sample_polynomial(self) -> MPolynomial_libsingular:
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

    def _generate_random_polynomial(
        self, degree: int, num_terms: int
    ) -> MPolynomial_libsingular:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == "uniform"

        R = self.get_ring()
        field = R.base_ring()

        # First, create a polynomial with all coefficients equal to 1
        ZZ_R = PolynomialRing(ZZ, R.gens(), order=R.term_order())
        p = ZZ_R.random_element(
            degree=degree, terms=num_terms, choose_degree=choose_degree, x=1, y=2
        )

        # Get the dictionary representation of the polynomial
        p_dict = p.dict()

        # Randomly sample coefficients for each term based on the appropriate field
        for k, v in p_dict.items():
            if field == QQ:
                bound = self.num_bound if self.num_bound is not None else 10
                # For QQ, generate numerator and denominator randomly
                if self.nonzero_coeff:
                    # Exclude zero by ensuring numerator is not zero
                    num = (
                        randint(1, bound)
                        if RR.random_element(0, 1) < 0.5
                        else randint(-bound, -1)
                    )
                else:
                    num = randint(-bound, bound)
                den = randint(1, bound)
                p_dict[k] = QQ(num) / QQ(den)
            elif field == RR:
                coeff = self.max_coeff if self.max_coeff is not None else 10
                if self.nonzero_coeff:
                    # Exclude zero by sampling from non-zero range
                    p_dict[k] = RR.random_element(min=-coeff, max=coeff)
                    # Ensure non-zero by regenerating if zero
                    while p_dict[k] == 0:
                        p_dict[k] = RR.random_element(min=-coeff, max=coeff)
                else:
                    p_dict[k] = RR.random_element(min=-coeff, max=coeff)
            elif field == ZZ:
                coeff = self.max_coeff if self.max_coeff is not None else 10
                if self.nonzero_coeff:
                    # Exclude zero by sampling from non-zero range
                    p_dict[k] = (
                        randint(1, coeff)
                        if RR.random_element(0, 1) < 0.5
                        else randint(-coeff, -1)
                    )
                else:
                    p_dict[k] = randint(-coeff, coeff)
            elif field.characteristic() > 0:
                # For finite fields, randomly select values from 0 to p-1
                field_order = field.characteristic()

                assert field.is_prime_field(), (
                    f"Non-prime field detected: {field}. This may cause unexpected behavior."
                )

                if self.nonzero_coeff:
                    # Exclude zero by sampling from 1 to p-1
                    p_dict[k] = field(randint(1, field_order - 1))
                else:
                    p_dict[k] = field(randint(0, field_order - 1))
            else:
                raise ValueError(f"Unsupported field: {field}")

        # breakpoint()
        # Convert to the original polynomial ring R
        return R(p_dict)

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
            p = self._sample_polynomial()
            # Apply density
            if RR.random_element(0, 1) >= density:
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


def compute_max_coefficient(poly: MPolynomial_libsingular) -> int:
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
