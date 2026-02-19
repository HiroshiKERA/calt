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
from sage.misc.prandom import sample
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular


class PolynomialSampler:
    """Generator for random polynomials with specific constraints.

    The sampler builds polynomials by first choosing a target degree and number
    of terms (within min/max bounds), then selecting that many distinct
    monomials and assigning random coefficients from the base ring. Ring and
    constraints can be given either as (symbols, field_str, order) or as a
    pre-built PolynomialRing.

    Behavior summary
    -----------------

    **degree_sampling** controls how monomial degrees are chosen:

    - ``'uniform'``: For each term, a degree in [min_degree, max_degree] is
      chosen uniformly at random, then a monomial of that degree is chosen.
      The resulting polynomial's degree distribution is more uniform over the
      range.
    - ``'fixed'``: Monomials are chosen uniformly from all monomials of degree
      at most max_degree. Because there are more such monomials at higher
      degrees, the polynomial tends to have total degree equal to max_degree.

    **Degree and number of terms**: Every returned polynomial has total
    degree >= min_degree. The guarantees on total degree and number of
    terms depend on ``strictly_conditioned`` and ``nonzero_coeff``; see
    the constructor parameters for details.
    """

    def __init__(
        self,
        symbols: str | None = None,
        field_str: str | None = None,
        order: str | TermOrder | None = "degrevlex",
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
        nonzero_coeff: bool = True,
        max_attempts: int = 1000,
    ):
        """
        Initialize polynomial sampler.

        Args:
            symbols: Variable names for the polynomial ring
                (required if ring is None).
            field_str: Base ring specifier: "QQ", "RR", "ZZ", or "GF(p)"
                for a prime finite field (required if ring is None).
            order: Term order of the ring, e.g. "degrevlex"
                (required if ring is None).
            ring: Pre-built PolynomialRing
                (alternative to symbols/field_str/order).
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
                in [1, max_num_terms]; ``'fixed'``: use max_num_terms.
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
                    - If nonzero_coeff=False, some polynomials have
                      fewer than num_terms terms (zero coefficients);
                      those are rejected and generation is retried.
                      RuntimeError is raised if no success within
                      max_attempts.
                - If False:
                    - Return the first polynomial with total degree >=
                      min_degree and (if nonzero_instance) non-zero.
                    - Number of terms may be less than the chosen
                      num_terms when nonzero_coeff=False.
            nonzero_instance: If True, the zero polynomial is never
                returned.
            nonzero_coeff: If True, no coefficient is zero (default);
                gives a predictable number of terms and fewer retries
                when strictly_conditioned is True.
            max_attempts: Maximum trials per polynomial when
                strictly_conditioned is True; RuntimeError is raised if
                no success.
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
            # Map "grevlex" to "degrevlex" for SageMath compatibility
            # SageMath uses "degrevlex" instead of "grevlex"
            if isinstance(order, str) and order == "grevlex":
                self.order = "degrevlex"
            else:
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
        """Convert field_str to the SageMath base ring (QQ, RR, ZZ, or GF(p))."""
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
        Return the polynomial ring (the configured ring if set, otherwise one built from symbols/field_str/order).

        Returns:
            PolynomialRing: The polynomial ring.

        Raises:
            ValueError: If polynomial ring creation fails with informative error message.
        """
        if self.ring is not None:
            return self.ring

        try:
            field = self.get_field()
            R = PolynomialRing(field, self.symbols, order=self.order)
            return R
        except (ValueError, TypeError, AttributeError) as e:
            # Provide informative error message with the parameters used
            field_str = self.field_str if self.field_str else "unknown"
            order_str = (
                str(self.order)
                if isinstance(self.order, (str, TermOrder))
                else self.order
            )
            raise ValueError(
                f"Failed to create polynomial ring with parameters: "
                f"field={field_str}, symbols={self.symbols}, order={order_str}. "
                f"Error details: {str(e)}"
            ) from e

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
        is_univariate = R.ngens() == 1
        for attempt in range(self.max_attempts):
            p = self._generate_random_polynomial(degree, num_terms, is_univariate)

            # Check conditions
            if p == 0 and self.nonzero_instance:
                continue

            # Univariate polynomials use degree(), multivariate use total_degree()
            poly_degree = p.degree() if is_univariate else p.total_degree()
            if poly_degree < self.min_degree:
                continue

            if not self.strictly_conditioned:
                break

            if poly_degree == degree and len(p.monomials()) == num_terms:
                break

            if attempt == self.max_attempts - 1:
                raise RuntimeError(
                    f"Failed to generate polynomial satisfying conditions after {self.max_attempts} attempts"
                )

        return p

    def _generate_univariate_polynomial_dict(
        self, R: PolynomialRing, ZZ_R: PolynomialRing, degree: int, num_terms: int
    ) -> dict:
        """
        Generate a dictionary representation of a univariate polynomial
        with specified degree and number of terms.

        Args:
            R: Original polynomial ring
            ZZ_R: Integer polynomial ring for generating structure
            degree: Maximum degree of the polynomial
            num_terms: Number of terms in the polynomial (already validated to be <= degree + 1)

        Returns:
            Dictionary representation of the polynomial (monomial -> coefficient)
        """
        var = R.gen(0)
        # Select random degrees for the terms (num_terms is already <= degree + 1)
        possible_degrees = list(range(degree + 1))
        selected_degrees = sample(possible_degrees, num_terms)

        # Create polynomial with coefficient 1 for selected terms
        p = ZZ_R(0)
        for d in selected_degrees:
            p += var**d

        return p.dict()

    def _generate_random_polynomial(
        self, degree: int, num_terms: int, is_univariate: bool
    ) -> MPolynomial_libsingular:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == "uniform"

        R = self.get_ring()
        field = R.base_ring()

        # First, create a polynomial with all coefficients equal to 1
        # For univariate rings, term_order() is not available
        if is_univariate:
            # Univariate case: order doesn't matter
            ZZ_R = PolynomialRing(ZZ, R.gens())
            # Univariate random_element doesn't support 'terms' parameter
            # Manually construct polynomial with specified number of terms
            p_dict = self._generate_univariate_polynomial_dict(
                R, ZZ_R, degree, num_terms
            )
        else:
            # Multivariate case: use term_order
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

        # Convert to the original polynomial ring R
        return R(p_dict)

    def _sample_matrix(
        self,
        size: tuple[int, int],
        density: float = 1.0,
        matrix_type: str | None = None,
    ) -> matrix:
        """
        Generate a matrix of random polynomials.

        Args:
            size: (rows, cols) shape of the matrix.
            density: Probability that each entry is non-zero (entries may be zeroed at random).
            matrix_type: If "unimodular_upper_triangular", set diagonal to 1 and strict lower part to 0.

        Returns:
            A matrix over the polynomial ring with entries sampled by this sampler.
        """
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


def compute_max_coefficient(poly: MPolynomial_libsingular) -> int | float:
    """Compute maximum absolute coefficient value in a polynomial. Returns int for QQ/ZZ, float-like for RR."""
    coeffs = poly.coefficients()
    field = poly.base_ring()

    if not coeffs:
        return 0

    if field == RR:
        return max(abs(c) for c in coeffs)
    else:  # QQ case
        return max(max(abs(c.numerator()), abs(c.denominator())) for c in coeffs)


def compute_matrix_max_coefficient(M: matrix) -> int | float:
    """Compute maximum absolute coefficient value over all entries of a polynomial matrix. Type as in compute_max_coefficient."""
    return max(compute_max_coefficient(p) for p in M.list())
