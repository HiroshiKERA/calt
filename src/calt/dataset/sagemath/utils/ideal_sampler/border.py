"""Direct sampling of border bases of zero-dimensional ideals (NeurIPS'25).

Reference
---------
Kera et al. (NeurIPS 2025). Original implementation: ``BorderBasisSampler`` /
``random_border_basis`` in
https://github.com/HiroshiKERA/OracleBorderBasis
(``src/border_basis_lib/border_basis_sampling.py``).

Idea
----
A border basis of a zero-dimensional ideal is determined by an *order ideal*
``O`` (a finite, divisor-closed set of monomials) and its *border* ``B`` (the
monomials ``x_i · o`` not already in ``O``). We:

  1. sample a random order ideal ``O`` (by recursively splitting the staircase),
  2. compute its border ``B``,
  3. draw ``|O|`` random evaluation points ``P`` so that the order-ideal
     monomials are linearly independent there, and
  4. solve a linear system (kernel of ``[B(P) | O(P)]``) to obtain the border
     basis ``G`` whose vanishing ideal at ``P`` has ``O`` as a basis of the
     quotient.

``BorderBasisSampler.sample`` returns the basis ``G`` (a list of ring elements);
``random_border_basis`` returns the full record (basis, border, order, points).
"""

# ``O`` (order ideal), ``B`` (border), ``V`` (kernel) follow the paper's math
# notation, so we allow ruff's "ambiguous variable name" rule here.
# ruff: noqa: E741
import itertools as it
from copy import deepcopy
from dataclasses import dataclass
from random import choice, shuffle

import numpy as np
from sage.all import QQ, Matrix, MatrixSpace, Partitions


# --------------------------------------------------------------------------- #
# Helpers (ported from border_basis_lib/utils.py)                              #
# --------------------------------------------------------------------------- #
def border(order_ideal):
    """Border of an order ideal: monomials ``x_i · o`` (o in O) not already in O."""
    span_variables = order_ideal[0].args()
    B = []
    for x in span_variables:
        B += [x * o for o in order_ideal if x * o not in order_ideal]
    return sorted(set(B))


def subs(F, P):
    """Evaluate polynomials ``F`` at the rows of point-matrix ``P``.

    Returns a ``(num_points x num_polys)`` matrix over the base field.
    """
    field = P[0, 0].base_ring()
    num_points = P.nrows()
    FP = [f(*p) for p, f in it.product(P, F)]
    return MatrixSpace(field, num_points, len(F))(FP)


def is_regular(M):
    """True iff ``M`` has full rank ``min(rows, cols)``."""
    return M.rank() == min(M.ncols(), M.nrows())


def keyword_for_numbound(field, bound):
    """Random-element kwargs bounding coefficient size (only meaningful over QQ)."""
    return {"num_bound": bound} if field == QQ else {}


# --------------------------------------------------------------------------- #
# Staircase geometry                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """An axis-aligned segment in n-dim space defined by two endpoints."""

    endpoints: list

    def __post_init__(self):
        self.lb = np.array(np.minimum.reduce(self.endpoints), dtype=int)
        self.ub = np.array(np.maximum.reduce(self.endpoints), dtype=int)
        self.n = len(self.lb)

    def __hash__(self):
        return hash((tuple(self.lb), tuple(self.ub)))


@dataclass
class NeighborSegments:
    """Collection of segments meeting at an intersecting point."""

    segments: list
    intersecting_point: np.ndarray

    def __post_init__(self):
        self.n = self.segments[0].n
        max_point = np.vstack([segment.ub for segment in self.segments])
        self.max_point = np.min(max_point + np.eye(self.n, dtype=int) * 100000, axis=0)
        self.valid = np.sum(
            self.max_point != self.intersecting_point
        ) > 1 and not np.all(self.max_point - self.intersecting_point <= 1)

    def sampling(self) -> np.ndarray:
        nondegenerated = np.array(self.intersecting_point < self.max_point + 1)
        point = self.intersecting_point.copy()
        point[nondegenerated] = np.random.randint(
            self.intersecting_point[nondegenerated], self.max_point[nondegenerated] + 1
        )
        return point

    def split_at(self, splitpoint: np.ndarray) -> list:
        new_segment = Segment(deepcopy([self.intersecting_point, splitpoint]))
        new_neighborsegments = []
        for i in range(self.n):
            new_segments = []
            for j, segment in enumerate(self.segments):
                if i != j:
                    lb = segment.lb
                    lb[j] = splitpoint[j]
                    ub = segment.ub
                    new_segments.append(Segment([lb, ub]))
                else:
                    new_segments.append(deepcopy(new_segment))
            new_intersecting_point = self.intersecting_point + np.eye(self.n)[i] * (
                splitpoint[i] - self.intersecting_point[i]
            )
            new_neighborsegments.append(
                NeighborSegments(deepcopy(new_segments), new_intersecting_point)
            )
        return new_neighborsegments


class BorderBasisSampler:
    """Sample border bases of zero-dimensional ideals (NeurIPS'25)."""

    def __init__(self, ring):
        self.ring = ring
        self.n = ring.ngens()

    # ---- order-ideal sampling -------------------------------------------- #
    def hypercube_points(self, u, v, exclude_max: bool = True) -> list:
        u, v = np.minimum(u, v), np.maximum(u, v)
        grid_ranges = [np.arange(u[i], v[i] + 1, dtype=int) for i in range(len(u))]
        return [
            tuple(map(int, p))
            for p in it.product(*grid_ranges)
            if not (exclude_max and np.array_equal(p, v))
        ]

    def span_order_ideal(self, neighbor_segments: list) -> list:
        order_ideal = []
        for ns in neighbor_segments:
            order_ideal.extend(
                deepcopy(
                    self.hypercube_points(
                        ns.intersecting_point, ns.max_point, exclude_max=False
                    )
                )
            )
        return order_ideal

    def sample_order_ideal(self, degree_bounds, max_iters: int = 100) -> list:
        origin = np.zeros(self.n, dtype=int)
        max_point = np.array(degree_bounds)

        S = []
        for i in range(self.n):
            endpoint = max_point.copy()
            endpoint[i] = 0
            S.append(Segment([origin, endpoint]))

        N = [NeighborSegments(deepcopy(S), origin)]
        T = []

        O_axis = []
        for i in range(self.n):
            canonical_basis = np.eye(self.n, dtype=int)[i]
            O_axis.extend(
                self.hypercube_points(
                    origin, canonical_basis * degree_bounds[i], exclude_max=False
                )
            )

        for i in range(max_iters):
            if i == max_iters - 1 or not N:
                break
            neighbor_segment = N.pop()
            splitpoint = neighbor_segment.sampling()
            new_neighborsegments = [
                ns for ns in neighbor_segment.split_at(splitpoint) if ns.valid
            ]
            neighbor_segment.max_point = splitpoint
            T.append(neighbor_segment)
            N.extend(new_neighborsegments)
            shuffle(N)

        O = list(set(O_axis + self.span_order_ideal(T)))
        O.sort(key=lambda x: (-sum(x), *reversed(x)))  # grevlex
        return O

    def random_order_ideal(
        self, degree_bounds, total_degree_bound=None, degree_lower_bounds=None
    ) -> list:
        if degree_lower_bounds is None:
            degree_lower_bounds = 0
        if total_degree_bound is None:
            upper_bounds = np.random.randint(
                degree_lower_bounds, np.array(degree_bounds) + 1
            )
        else:
            random_total_degree = np.random.randint(1, total_degree_bound + 1)
            partitions = list(
                Partitions(
                    random_total_degree, max_length=self.n, max_part=max(degree_bounds)
                )
            )
            partition = list(choice(partitions))
            partition = partition + [0] * (self.n - len(partition))
            shuffle(partition)
            upper_bounds = partition
        return self.sample_order_ideal(upper_bounds)

    # ---- border basis from order ideal ----------------------------------- #
    def compute_border_basis(self, B, O, P):
        ring = self.ring
        O = [ring(o) for o in O]
        OP = subs(O, P)
        if not is_regular(OP):
            return None, False
        B = [ring(b) for b in B]
        BP = subs(B, P)
        M = BP.augment(OP)
        V = M.transpose().kernel().basis()
        return V, True

    def random_border_basis(
        self,
        degree_bounds,
        max_sampling: int = 100,
        total_degree_bound=None,
        degree_lower_bounds=None,
    ) -> dict:
        assert len(degree_bounds) == self.n
        ring = self.ring

        tdb = None if total_degree_bound is None else total_degree_bound - 1
        O = self.random_order_ideal(
            np.array(degree_bounds) - 1,
            total_degree_bound=tdb,
            degree_lower_bounds=degree_lower_bounds,
        )
        O = [ring.monomial(*o) for o in O]
        B = border(O)
        B_exponents = sorted(
            (t.exponents()[0] for t in B), key=lambda x: (-sum(x), *reversed(x))
        )
        B = [ring.monomial(*e) for e in B_exponents]

        MSpace = MatrixSpace(ring.base_ring(), len(O), self.n)
        success = False
        P = None
        for i in range(max_sampling):
            if i == max_sampling - 1:
                break
            P = MSpace.random_element(**keyword_for_numbound(ring.base_ring(), 10))
            V, success = self.compute_border_basis(B, O, P)
            if success:
                break

        G = []
        if success:
            G = (Matrix(B + O) * Matrix(V).T)[0]
            G = [ring(g) for g in G]

        return {
            "basis": G,
            "order_coeff": V if success else None,
            "border": B,
            "order": O,
            "points": P,
            "success": success,
        }

    def sample(
        self,
        num_samples: int,
        degree_bounds,
        total_degree_bound=None,
        max_sampling: int = 100,
    ) -> list:
        """Return ``num_samples`` border bases (each a list of ring elements).

        Samples are retried until a valid border basis is found.
        """
        out = []
        while len(out) < num_samples:
            rec = self.random_border_basis(
                degree_bounds,
                max_sampling=max_sampling,
                total_degree_bound=total_degree_bound,
            )
            if rec["success"]:
                out.append(rec["basis"])
        return out
