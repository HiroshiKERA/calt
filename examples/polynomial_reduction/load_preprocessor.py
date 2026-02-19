"""Load preprocessor for polynomial reduction: grevlex/lex, pattern 1 (remainder only) or 2 (quotients + remainder).

Data is stored as (f, G), (r,). For pattern 2, quotients are computed at load time via (f - r).lift(I).
"""

from typing import Any

from calt.io.preprocessor.load_preprocessor import _get_answer_from_source


class PolynomialReductionLoadPreprocessor:
    """Convert (f, G), (r,) to (input_text, target_text).

    - order: "grevlex" (use as-is) or "lex" (convert f to lex ring, G via FGLM to lex GB).
    - pattern: 1 = target is remainder only; 2 = target is quotients | remainder (quotients computed via lift).
    """

    INNER_SEP = " | "

    def __init__(self, order: str = "grevlex", pattern: int = 1):
        self.order = order
        self.pattern = pattern

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError("PolynomialReductionLoadPreprocessor expects dict source")
        problem = source.get("problem")
        answer = _get_answer_from_source(source)
        if problem is None or answer is None:
            raise ValueError(
                "Source must have 'problem' and 'answer' (or 'solution') keys"
            )
        (f, G) = problem
        (r,) = answer

        if self.order == "lex":
            f, G = _to_lex(f, G)
            r, quotients = _to_lex_remainder_quotients(f, G)
        else:
            quotients = _get_quotients(f, r, G) if self.pattern == 2 else None

        input_parts = [str(f)] + [str(g) for g in G]
        input_text = self.INNER_SEP.join(input_parts)

        if self.pattern == 1:
            target_text = str(r)
        else:
            q_parts = [str(q) for q in quotients] + [str(r)]
            target_text = self.INNER_SEP.join(q_parts)

        return input_text, target_text


def _get_quotients(f, r, G):
    """Return the list of quotients via (f - r).lift(I), where I = R.ideal(G)."""
    R = f.parent()
    ideal = R.ideal(list(G))
    try:
        q = (f - r).lift(ideal)
        return tuple(q) if q is not None and len(q) == len(G) else (R.zero(),) * len(G)
    except Exception:
        return (R.zero(),) * len(G)


def _to_lex(f, G):
    """Convert f and G from grevlex ring to lex: f by change_ring, G by FGLM."""
    R_grevlex = f.parent()
    from sage.rings.polynomial.polynomial_ring import is_PolynomialRing

    if not is_PolynomialRing(R_grevlex):
        return f, G
    names = R_grevlex.variable_names()
    from sage.all import PolynomialRing

    # FGLM requires zero-dimensional ideal; use same base ring as f
    base = R_grevlex.base_ring()
    R_lex = PolynomialRing(base, names, order="lex")
    f_lex = R_lex(f)
    ideal = R_grevlex.ideal(list(G))
    try:
        G_lex = ideal.transformed_basis("fglm", R_lex)
    except (ValueError, TypeError, NotImplementedError):
        # Fallback: just change ring (not a lex GB)
        G_lex = [R_lex(g) for g in G]
    return f_lex, tuple(G_lex)


def _to_lex_remainder_quotients(f_lex, G_lex):
    """Recompute remainder and quotients of f_lex by G_lex in lex ring."""
    R_lex = G_lex[0].parent()
    I_lex = R_lex.ideal(list(G_lex))
    r_lex = I_lex.reduce(f_lex)
    try:
        q_lex = (f_lex - r_lex).lift(I_lex)
        if q_lex is None or len(q_lex) != len(G_lex):
            q_lex = tuple(R_lex.zero() for _ in G_lex)
        else:
            q_lex = tuple(q_lex)
    except Exception:
        q_lex = tuple(R_lex.zero() for _ in G_lex)
    return r_lex, q_lex
