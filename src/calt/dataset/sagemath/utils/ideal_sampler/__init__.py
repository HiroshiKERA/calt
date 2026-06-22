"""Direct samplers of ideal bases ("backward" data generation).

Instead of sampling a generating set F and *computing* a basis G (the forward
direction), these samplers draw a basis G directly, following the algorithms in:

  - Gröbner bases in shape position — Kera et al., "Learning to Compute Gröbner
    Bases" (NeurIPS 2024). See :class:`GroebnerBasisSampler`.
  - Border bases — Kera et al. (NeurIPS 2025). See :class:`BorderBasisSampler`.

Pair these with an ideal-invariant transform (``ideal_invariant_transformer``)
to obtain a generating set F of the same ideal, giving a training pair (F, G).
"""

from .border import BorderBasisSampler
from .groebner import GroebnerBasisSampler

__all__ = ["GroebnerBasisSampler", "BorderBasisSampler"]
