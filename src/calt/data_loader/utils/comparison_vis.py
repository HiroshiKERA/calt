from typing import Mapping, Sequence, Tuple
import re
from sympy import Expr, Poly, Symbol, latex
from IPython.display import display, Math


__all__ = ["display_with_diff"]

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _poly_to_dict(poly: Poly) -> dict[Tuple[int, ...], int]:
    """Return a mapping {exponent_tuple: coefficient}."""
    return {e: int(c) for e, c in poly.terms()}


# ---------------------------------------------------------------------------
# Monomial → LaTeX -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _term_latex(
    coeff: int,
    exps: Sequence[int],
    var_syms: Sequence[Symbol],
    *,
    highlight: bool = False,
    highlight_coeff_only: bool = False,
) -> str:
    """Return LaTeX for one monomial with optional highlighting."""

    # --- sign & absolute coefficient ------------------------------------- #
    sign = "-" if coeff < 0 else ""
    abs_coeff = abs(coeff)
    has_vars = any(e != 0 for e in exps)

    coeff_str = "" if abs_coeff == 1 and has_vars else str(abs_coeff)

    # --- variable part ---------------------------------------------------- #
    var_parts: list[str] = []
    for v, e in zip(var_syms, exps):
        if e == 0:
            continue
        elif e == 1:
            var_parts.append(latex(v))
        else:
            var_parts.append(f"{latex(v)}^{{{e}}}")

    # Combine: coefficient  gap  variables
    body = coeff_str + (r"\, " if coeff_str and var_parts else "") + r"\, ".join(var_parts) or "0"
    term_tex = sign + body

    # --- highlighting ----------------------------------------------------- #
    if highlight:
        term_tex = rf"\cancel{{{term_tex}}}"
    elif highlight_coeff_only and coeff_str:
        term_tex = re.sub(
            re.escape(coeff_str),
            lambda _: rf"\cancel{{{coeff_str}}}",
            term_tex,
            count=1,
        )
    return term_tex


# ---------------------------------------------------------------------------
# Build full polynomial LaTeX ------------------------------------------------
# ---------------------------------------------------------------------------


def _build_poly_latex(
    poly_dict: Mapping[Tuple[int, ...], int],
    var_syms: Sequence[Symbol],
    diff_info: Mapping[Tuple[int, ...], str],
) -> str:
    """Return LaTeX string for *predicted* polynomial with diff marks."""

    tex_terms: list[str] = []

    for exps in sorted(poly_dict.keys(), reverse=True):  # deterministic order
        coeff = poly_dict[exps]
        if coeff == 0:
            continue

        diff_type = diff_info.get(exps, "")  # "extra", "coeff_wrong", or ""
        hl_all = diff_type == "extra"
        hl_coeff = diff_type == "coeff_wrong"

        term_tex = _term_latex(
            coeff,
            exps,
            var_syms,
            highlight=hl_all,
            highlight_coeff_only=hl_coeff,
        )

        # keep explicit sign for first term if negative, else prepend +
        if term_tex.startswith("-"):
            tex_terms.append(term_tex)
        else:
            tex_terms.append("+" + term_tex if tex_terms else term_tex)

    return " ".join(tex_terms) if tex_terms else "0"


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def display_with_diff(
    gold: Expr,
    pred: Expr,
    var_order: Sequence[Symbol] | None = None,
) -> None:
    """Render *gold* vs. *pred* with strikethrough on mistakes in *pred*.

    Parameters
    ----------
    gold, pred : sympy.Expr
        Ground‑truth and model‑predicted expressions.
    var_order : list[sympy.Symbol] | None
        Variable ordering (important for >2 variables). Inferred if None.
    """

    # --- normalize -------------------------------------------------------- #
    if var_order is None:
        var_order = sorted(gold.free_symbols.union(pred.free_symbols), key=lambda s: s.name)
    gold_poly = Poly(gold.expand(), *var_order)
    pred_poly = Poly(pred.expand(), *var_order)

    gdict = _poly_to_dict(gold_poly)
    pdict = _poly_to_dict(pred_poly)

    # --- diff detection --------------------------------------------------- #
    diff: dict[Tuple[int, ...], str] = {}
    for exps in set(gdict) | set(pdict):
        gcoeff = gdict.get(exps, 0)
        pcoeff = pdict.get(exps, 0)
        if pcoeff == 0 and gcoeff != 0:
            continue  # missing term (not highlighted)
        if gcoeff == 0 and pcoeff != 0:
            diff[exps] = "extra"
        elif gcoeff != pcoeff:
            diff[exps] = "coeff_wrong"

    # --- render ----------------------------------------------------------- #
    gold_tex = latex(gold.expand())
    pred_tex = _build_poly_latex(pdict, var_order, diff)

    display(Math(r"\text{Gold:}\; " + gold_tex))
    display(Math(r"\text{Predicted:}\; " + pred_tex))
