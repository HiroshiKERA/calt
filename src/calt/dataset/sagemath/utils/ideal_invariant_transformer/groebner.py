"""Ideal-invariant transform for Gröbner-basis data (NeurIPS'24).

Reference
---------
Kera et al., "Learning to Compute Gröbner Bases" (NeurIPS 2024). Original
implementation: ``random_non_gb`` in
https://github.com/HiroshiKERA/transformer-groebner (``src/dataset/groebner.sage``).

Given a basis ``G`` (as an ``n x 1`` column over the polynomial ring), draw a
random ideal-preserving transform

    F = U · P · A · G

where ``A`` is an ``m x n`` unimodular upper-triangular polynomial matrix
(``m >= n``), ``U`` is an ``m x m`` unimodular upper-triangular polynomial
matrix, and ``P`` is an ``m x m`` permutation matrix. The result ``F`` is a
generating set of ``m`` polynomials with ``<F> = <G>`` but, in general, ``F`` is
not a Gröbner basis — exactly the (F, G) training pair the model learns from.
"""

from sage.all import Permutation, Permutations, matrix, randint

from ..polynomial_sampler import PolynomialSampler, compute_max_coefficient


def random_permutation_matrix(m: int):
    """Random ``m x m`` permutation matrix (SageMath)."""
    perm = Permutations(list(range(1, m + 1))).random_element()
    return matrix(Permutation(perm))


def _max_abs_coeff_in_matrix(M) -> float:
    """Largest absolute coefficient appearing in any polynomial entry of ``M``."""
    best = 0
    for entry in M.list():
        if entry != 0:
            best = max(best, compute_max_coefficient(entry))
    return best


def _unimodular(sampler, rows: int, cols: int, density: float):
    """Reference-faithful unimodular matrix (transformer-groebner convention).

    1 on the diagonal, 0 strictly above it, random below. For a rectangular
    ``rows > cols`` matrix the extra bottom rows lie below the diagonal and so
    stay random — unlike zeroing the strict-lower part, which would make those
    rows all-zero and produce zero generators in ``F``.
    """
    M = sampler.sample(num_samples=1, size=(rows, cols), density=density)[0]
    for i in range(rows):
        for j in range(cols):
            if i == j:
                M[i, j] = 1
            elif i < j:
                M[i, j] = 0
    return M


class GroebnerIdealTransformer:
    """Apply the Bruhat-decomposition ideal-invariant transform ``F = U·P·A·G``.

    Args:
        ring: SageMath multivariate ``PolynomialRing`` (must match ``G``).
        max_size: Maximum number of generators in ``F`` (``>= n = len(G)``).
        max_degree, min_degree, max_num_terms, max_coeff, num_bound,
        degree_sampling, term_sampling, strictly_conditioned: forwarded to the
            ``PolynomialSampler`` used for the random matrices.
        density: Probability of non-zero off-diagonal entries in the matrices.
        coeff_bound: Over infinite fields (QQ/ZZ/RR), retry while the largest
            coefficient in ``F`` exceeds this bound (coefficient-swell guard).
        max_iter: Maximum number of retries for the coefficient bound.
    """

    def __init__(
        self,
        ring,
        max_size: int,
        max_degree: int = 1,
        min_degree: int = 1,
        max_num_terms: int | None = None,
        max_coeff: int | None = None,
        num_bound: int | None = None,
        degree_sampling: str = "uniform",
        term_sampling: str = "uniform",
        strictly_conditioned: bool = True,
        density: float = 1.0,
        coeff_bound: int = 100,
        max_iter: int = 100,
    ):
        self.ring = ring
        self.max_size = max_size
        self.density = density
        self.coeff_bound = coeff_bound
        self.max_iter = max_iter
        self._sampler = PolynomialSampler(
            ring=ring,
            order=None,  # PolynomialSampler forbids ring + order together
            max_degree=max_degree,
            min_degree=min_degree,
            max_num_terms=max_num_terms,
            max_coeff=max_coeff,
            num_bound=num_bound,
            degree_sampling=degree_sampling,
            term_sampling=term_sampling,
            strictly_conditioned=strictly_conditioned,
            nonzero_instance=True,
        )

    def transform(self, G: list) -> list:
        """Return a generating set ``F`` (list of ring elements) with ``<F> = <G>``."""
        ring = self.ring
        n = len(G)
        SG = matrix(ring, n, 1, list(G))
        is_finite = ring.base_ring().is_finite()

        F = None
        for _ in range(self.max_iter):
            m = randint(0, self.max_size - n) + n
            A = _unimodular(self._sampler, m, n, self.density)
            U = _unimodular(self._sampler, m, m, self.density)
            P = random_permutation_matrix(m)

            F = U * P * A * SG

            if is_finite or _max_abs_coeff_in_matrix(F) <= self.coeff_bound:
                break

        return [F[i, 0] for i in range(F.nrows())]
