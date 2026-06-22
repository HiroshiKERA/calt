"""Direct sampling of Gröbner bases of ideals in shape position (NeurIPS'24).

Reference
---------
Kera, Ishihara, Kambe, Vaccon, Yokoyama, "Learning to Compute Gröbner Bases"
(NeurIPS 2024). Original implementation: ``random_shape_gb`` in
https://github.com/HiroshiKERA/transformer-groebner (``src/dataset/groebner.sage``).

Idea
----
Rather than sampling a generating set ``F`` and running Buchberger to obtain a
Gröbner basis ``G`` (the forward direction implemented elsewhere in CALT), we
sample ``G`` directly. An ideal is in *shape position* when its reduced Gröbner
basis w.r.t. the lex order ``x_0 > x_1 > ... > x_{n-1}`` has the form

    G = [ x_0 - g_1(x_{n-1}),
          x_1 - g_2(x_{n-1}),
          ...
          x_{n-2} - g_{n-1}(x_{n-1}),
          h(x_{n-1}) ]

where ``h`` is a monic univariate polynomial in the last variable and each
``g_i`` is a univariate polynomial of degree ``< deg(h)``. Such a ``G`` is, by
construction, a reduced Gröbner basis (each ``x_i`` is the leading term and the
tail involves only ``x_{n-1}``), so no Gröbner computation is needed.
"""

from sage.all import PolynomialRing, matrix

from ..polynomial_sampler import PolynomialSampler


def _to_monic(p):
    """Scale ``p`` so its leading coefficient is 1 (``p`` is over a field)."""
    return p / p.lc()


class GroebnerBasisSampler:
    """Sample reduced Gröbner bases of ideals in shape position.

    Args:
        ring: SageMath multivariate ``PolynomialRing`` (the target ring). The lex
            shape position is taken w.r.t. its variable order.
        max_degree: Maximum degree of the univariate tail ``h``.
        min_degree: Minimum degree of ``h`` (``>= 1``).
        max_num_terms: Upper bound on the number of terms per univariate poly.
        max_coeff: Coefficient bound for RR/ZZ (passed to ``PolynomialSampler``).
        num_bound: Numerator/denominator bound for QQ.
        degree_sampling, term_sampling: ``'uniform'`` or ``'fixed'``.
        strictly_conditioned: Forwarded to ``PolynomialSampler``.
    """

    def __init__(
        self,
        ring,
        max_degree: int = 5,
        min_degree: int = 1,
        max_num_terms: int | None = None,
        max_coeff: int | None = None,
        num_bound: int | None = None,
        degree_sampling: str = "uniform",
        term_sampling: str = "uniform",
        strictly_conditioned: bool = True,
    ):
        self.ring = ring
        self.max_degree = max_degree
        self.min_degree = max(1, min_degree)
        self.max_num_terms = max_num_terms
        self.max_coeff = max_coeff
        self.num_bound = num_bound
        self.degree_sampling = degree_sampling
        self.term_sampling = term_sampling
        self.strictly_conditioned = strictly_conditioned

    def _usampler(self, uring, max_degree, min_degree):
        return PolynomialSampler(
            ring=uring,
            order=None,  # required: PolynomialSampler forbids ring + order together
            max_degree=max_degree,
            min_degree=min_degree,
            max_num_terms=self.max_num_terms,
            max_coeff=self.max_coeff,
            num_bound=self.num_bound,
            degree_sampling=self.degree_sampling,
            term_sampling=self.term_sampling,
            strictly_conditioned=self.strictly_conditioned,
            nonzero_instance=True,
        )

    def _sample_one(self) -> list:
        ring = self.ring
        field = ring.base_ring()
        x = ring.gens()
        n = ring.ngens()

        # Univariate ring in the last variable (lex), used to draw h and the g_i.
        uring = PolynomialRing(field, names=[str(x[-1])], order="lex")

        # 1) h: monic univariate of degree in [min_degree, max_degree].
        h = self._usampler(uring, self.max_degree, self.min_degree).sample(1)[0]
        h = _to_monic(h)
        deg_h = int(h.degree())

        # 2) g_1, ..., g_{n-1}: univariate of degree < deg(h).
        if n > 1:
            g_sampler = self._usampler(uring, max(0, deg_h - 1), 0)
            tail = g_sampler.sample(num_samples=1, size=(n - 1, 1))[0]  # (n-1) x 1
            G = tail.stack(matrix(uring, 1, 1, [h])).change_ring(ring)
        else:
            G = matrix(ring, 1, 1, [ring(h)])

        # Add the leading variables: row i (< n-1) becomes x_i + (univariate tail).
        X = matrix(ring, n, 1, [*x[:-1], 0])
        G = G + X
        return [G[i, 0] for i in range(n)]

    def sample(self, num_samples: int = 1) -> list:
        """Return ``num_samples`` Gröbner bases, each a list of ring elements."""
        return [self._sample_one() for _ in range(num_samples)]
