"""Tests for "backward" ideal data generation (sample basis G, transform to F).

Gröbner (NeurIPS'24): shape-position GB sampler + Bruhat transform F=U·P·A·G.
Border  (NeurIPS'25): border-basis sampler + generalized transform F=A·G.

Both directions are checked for ideal invariance (``<F> = <G>``) and, for
Gröbner, that ``G`` is exactly the reduced Gröbner basis of ``<F>``.
"""

import pytest

sage = pytest.importorskip("sage.all")
from sage.all import GF, QQ, Ideal, PolynomialRing  # noqa: E402

from calt.dataset.sagemath.utils.ideal_invariant_transformer import (  # noqa: E402
    BorderIdealTransformer,
    GroebnerIdealTransformer,
)
from calt.dataset.sagemath.utils.ideal_sampler import (  # noqa: E402
    BorderBasisSampler,
    GroebnerBasisSampler,
)


def _gb(ideal):
    """Gröbner basis via Singular's ``std`` (avoids a Hilbert-driven crash on
    some Singular builds with the default ``groebner`` algorithm)."""
    return list(ideal.groebner_basis(algorithm="singular:std"))


def _monic_set(polys):
    return {str(p / p.lc()) for p in polys}


def _same_ideal(F, G):
    return _monic_set(_gb(Ideal(F))) == _monic_set(_gb(Ideal(G)))


# --------------------------------------------------------------------------- #
# Gröbner (NeurIPS'24)                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", [GF(7), QQ])
def test_groebner_sampler_returns_groebner_bases(field):
    R = PolynomialRing(field, names=("x", "y", "z"), order="lex")
    sampler = GroebnerBasisSampler(
        R, max_degree=3, min_degree=1, max_num_terms=4, max_coeff=3, num_bound=3
    )
    for G in sampler.sample(5):
        assert len(G) == R.ngens()
        assert Ideal(G).basis_is_groebner() is True


@pytest.mark.parametrize("field", [GF(7), QQ])
def test_groebner_backward_preserves_ideal_and_target(field):
    R = PolynomialRing(field, names=("x", "y", "z"), order="lex")
    sampler = GroebnerBasisSampler(
        R, max_degree=3, min_degree=1, max_num_terms=4, max_coeff=3, num_bound=3
    )
    transformer = GroebnerIdealTransformer(
        R,
        max_size=6,
        max_degree=1,
        min_degree=0,
        max_num_terms=3,
        max_coeff=3,
        num_bound=3,
        coeff_bound=10**6,
    )
    for G in sampler.sample(5):
        F = transformer.transform(G)
        assert R.ngens() <= len(F) <= 6
        gF = _gb(Ideal(F))
        # <F> = <G>, and G is exactly the reduced Gröbner basis of <F>.
        assert all(f.reduce(G) == 0 for f in F)  # <F> ⊆ <G>
        assert all(g.reduce(gF) == 0 for g in G)  # <G> ⊆ <F>
        assert _monic_set(gF) == _monic_set(G)


# --------------------------------------------------------------------------- #
# Border (NeurIPS'25)                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", [GF(7), QQ])
def test_border_sampler_returns_zero_dimensional_bases(field):
    R = PolynomialRing(field, names=("x", "y"), order="degrevlex")
    sampler = BorderBasisSampler(R)
    Gs = sampler.sample(4, degree_bounds=[3, 3], total_degree_bound=4)
    assert len(Gs) == 4
    for G in Gs:
        assert len(G) >= 1
        assert Ideal(G).dimension() == 0  # border bases are 0-dimensional


@pytest.mark.parametrize("field", [GF(7), QQ])
def test_border_backward_preserves_ideal(field):
    """Faithful reference transform (random A) with constant (degree-0) entries
    is exactly ideal-invariant: F is a set of field-linear combinations of G."""
    R = PolynomialRing(field, names=("x", "y"), order="degrevlex")
    sampler = BorderBasisSampler(R)
    transformer = BorderIdealTransformer(
        R, max_size=6, max_degree=0, min_degree=0, max_coeff=3, num_bound=3
    )
    for G in sampler.sample(4, degree_bounds=[3, 3], total_degree_bound=4):
        F = transformer.transform(G)
        assert len(F) > len(G)
        assert _same_ideal(F, G)


def test_border_transform_identity_block_is_exact():
    """identity_block=True makes <F>=<G> exact for any entry degree (F contains G)."""
    R = PolynomialRing(GF(7), names=("x", "y"), order="degrevlex")
    G = BorderBasisSampler(R).sample(1, degree_bounds=[3, 3], total_degree_bound=4)[0]
    F = BorderIdealTransformer(
        R, max_size=5, identity_block=True, max_degree=2
    ).transform(G)
    assert all(g in F for g in G)  # the identity block keeps G verbatim in F
