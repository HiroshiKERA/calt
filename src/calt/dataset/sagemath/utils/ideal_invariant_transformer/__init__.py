"""Ideal-invariant transforms ("backward" data generation, step ii).

Given a basis ``G`` of an ideal ``I`` (e.g. sampled by ``ideal_sampler``), these
transforms produce a generating set ``F`` of the *same* ideal ``I = <F> = <G>``,
yielding a training pair ``(F, G)`` whose target ``G`` is the basis of ``<F>``.

  - Gröbner (NeurIPS'24): ``F = U · P · A · G`` with unimodular upper-triangular
    ``U, A`` and a permutation ``P`` (Bruhat-decomposition transform).
    See :class:`GroebnerIdealTransformer`.
  - Border (NeurIPS'25): ``F = A · G`` with a random matrix ``A`` — a generalized
    version of the above. See :class:`BorderIdealTransformer`.
"""

from .border import BorderIdealTransformer
from .groebner import GroebnerIdealTransformer

__all__ = ["GroebnerIdealTransformer", "BorderIdealTransformer"]
