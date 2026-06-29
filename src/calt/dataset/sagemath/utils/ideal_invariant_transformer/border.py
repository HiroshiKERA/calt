"""Ideal-invariant transform for border-basis data (NeurIPS'25).

Reference
---------
Kera et al. (NeurIPS 2025). Original implementation:
``backward_basis_transformation`` in
https://github.com/HiroshiKERA/OracleBorderBasis
(``src/dataset/generators/border_basis_generator.py``).

This is the *generalized* version of the Gröbner Bruhat transform: given a basis
``G`` (an ``n x 1`` column over the polynomial ring), draw a random ``m x n``
matrix ``A`` (``m > n``) and set

    F = A · G,

a generating set of ``m`` polynomials.

Default (``identity_block=False``) reproduces the reference exactly: ``A`` is
sampled fully at random. With **constant** (degree-0) entries this is exactly an
ideal-invariant transform — ``F`` is then a set of field-linear combinations of
the border basis, so ``<F> = <G>`` (empirically 100%). With higher-degree
entries the invariance is only generic (~90-95%). Set ``identity_block=True`` to
pin the top ``n`` rows of ``A`` to the identity, which makes ``<F> = <G>`` exact
for any entry degree (``F`` then contains ``G`` verbatim plus random combinations).
"""

from sage.all import identity_matrix, matrix, randint

from ..polynomial_sampler import PolynomialSampler, compute_max_coefficient


def _max_abs_coeff_in_matrix(M) -> float:
    best = 0
    for entry in M.list():
        if entry != 0:
            best = max(best, compute_max_coefficient(entry))
    return best


class BorderIdealTransformer:
    """Apply the generalized ideal-invariant transform ``F = A · G``.

    Args:
        ring: SageMath ``PolynomialRing`` (must match ``G``).
        max_size: Maximum number of generators in ``F``.
        identity_block: If False (default), reproduces the reference: ``A`` is
            fully random (exactly ideal-invariant with degree-0/constant entries;
            generic otherwise). If True, the first ``n`` rows of ``A`` are the
            identity so that ``<F> = <G>`` holds exactly for any entry degree.
        max_degree, min_degree, max_num_terms, max_coeff, num_bound,
        degree_sampling, term_sampling, strictly_conditioned: forwarded to the
            ``PolynomialSampler`` used for the rows of ``A`` (``max_degree=0``
            gives the exactly-invariant constant-coefficient transform).
        density, coeff_bound, max_iter: as in ``GroebnerIdealTransformer``.
    """

    def __init__(
        self,
        ring,
        max_size: int,
        identity_block: bool = False,
        max_degree: int = 1,
        min_degree: int = 0,
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
        self.identity_block = identity_block
        self.density = density
        self.coeff_bound = coeff_bound
        self.max_iter = max_iter
        self._sampler = PolynomialSampler(
            ring=ring,
            order=None,
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
            # m = target number of generators of F (m > n), as in the reference.
            m = randint(0, max(0, self.max_size - n)) + n + 1
            if self.identity_block:
                # Top n rows = identity (keep G verbatim), then m - n random rows.
                extra = self._sampler.sample(
                    num_samples=1, size=(m - n, n), density=self.density
                )[0]
                A = identity_matrix(ring, n).stack(extra)
            else:
                # Reference: A is a fully random m x n matrix.
                A = self._sampler.sample(
                    num_samples=1, size=(m, n), density=self.density
                )[0]
            F = A * SG

            if is_finite or _max_abs_coeff_in_matrix(F) <= self.coeff_bound:
                break

        return [F[i, 0] for i in range(F.nrows())]
