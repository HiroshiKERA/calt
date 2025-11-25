import json
import re
from typing import Mapping, Sequence

from IPython.display import Math, display
from sympy import Expr, Integer, Poly, Symbol, latex, symbols
from sympy.parsing.sympy_parser import parse_expr

__all__ = ["display_with_diff", "load_eval_results", "parse_poly"]

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _poly_to_dict(poly: Poly) -> dict[tuple[int, ...], int]:
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
    body = (
        coeff_str + (r"\, " if coeff_str and var_parts else "") + r"\, ".join(var_parts)
        or "0"
    )
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
    poly_dict: Mapping[tuple[int, ...], int],
    var_syms: Sequence[Symbol],
    diff_info: Mapping[tuple[int, ...], str],
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
    gold: Expr | str,
    pred: Expr | str,
    var_order: Sequence[Symbol] | None = None,
) -> None:
    """Render "gold" vs. "pred" with strikethrough on mistakes in "pred".

    Args:
        gold (sympy.Expr | str):
            Ground-truth expression. If a string, it will be parsed as a token
            sequence (e.g., "C1 E1 E1 C-3 E0 E7") via ``parse_poly``.
        pred (sympy.Expr | str):
            Model-predicted expression. If a string, it will be parsed as a token
            sequence via ``parse_poly``.
        var_order (Sequence[sympy.Symbol] | None, optional):
            Variable ordering (important for >2 variables). Inferred if None. Also
            passed to ``parse_poly`` if inputs are strings. Defaults to None.
    """

    # --- input conversion ------------------------------------------------- #
    if isinstance(gold, str):
        gold = parse_poly(gold, var_names=var_order)
    if isinstance(pred, str):
        pred = parse_poly(pred, var_names=var_order)

    # --- normalize -------------------------------------------------------- #
    if var_order is None:
        var_order = sorted(
            gold.free_symbols.union(pred.free_symbols), key=lambda s: s.name
        )
    gold_poly = Poly(gold.expand(), *var_order)
    pred_poly = Poly(pred.expand(), *var_order)

    gdict = _poly_to_dict(gold_poly)
    pdict = _poly_to_dict(pred_poly)

    # --- diff detection --------------------------------------------------- #
    diff: dict[tuple[int, ...], str] = {}
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

    display(
        Math(
            r"""\begin{aligned}
        \text{Ground truth\,:}\; & {}"""
            + gold_tex
            + r"""\\
        \text{Prediction\,:}\;   & {}"""
            + pred_tex
            + r"""
        \end{aligned}"""
        )
    )


def load_eval_results(file_path: str) -> tuple[list[str], list[str]]:
    """Load evaluation results from a JSON file.

    The JSON file should contain a list of objects with "generated" and "reference" keys.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - List of generated texts.
            - List of reference texts.
    """
    generated_texts = []
    reference_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        generated_texts.append(item.get("generated", ""))
        reference_texts.append(item.get("reference", ""))

    return generated_texts, reference_texts


def _parse_poly_from_tokens(
    tokens: str, var_names: Sequence[str | Symbol] | None = None
) -> Expr:
    """Convert an internal token sequence into a SymPy polynomial.

    For example: ``"C1 E1 E1 C-3 E0 E7"``.

    Args:
        tokens (str):
            Whitespace-separated string where a token starting with ``C`` indicates
            a coefficient and the following ``E`` tokens indicate exponents.
        var_names (Sequence[str | sympy.Symbol] | None, optional):
            Variable names (either strings or pre-created Symbol objects). If
            ``None`` (default), variables are auto-generated as x0, x1, …

    Returns:
        sympy.Expr: A SymPy expression corresponding to the polynomial.

    Raises:
        ValueError: If the token sequence is malformed or the number of variables does not
            match ``var_names``.
    """
    parts = tokens.strip().split()
    if not parts or not parts[0].startswith("C"):
        raise ValueError("Token sequence must start with a 'C' coefficient token.")

    # --- Infer the number of variables from the first term ---------------- #
    try:
        # Find the **index** of the first 'C' token after the initial one
        next_c_idx = next(
            idx for idx, p in enumerate(parts[1:], start=1) if p.startswith("C")
        )
    except StopIteration:
        # Single-term polynomial → treat end of list as “next C” position
        next_c_idx = len(parts)

    n_vars = next_c_idx - 1
    if n_vars <= 0:
        raise ValueError(
            "Malformed token sequence: need at least one exponent token "
            f"before the next 'C'; got n_vars={n_vars}."
        )

    # --- Prepare SymPy symbols ------------------------------------------- #
    if var_names is None:
        vars_ = symbols(" ".join(f"x{i}" for i in range(n_vars)))
    else:
        if len(var_names) != n_vars:
            raise ValueError(
                f"Expected {n_vars} variable name(s), got {len(var_names)}."
            )
        if all(isinstance(v, str) for v in var_names):
            vars_ = symbols(" ".join(var_names))
        elif all(isinstance(v, Symbol) for v in var_names):
            vars_ = tuple(var_names)
        else:
            raise TypeError("var_names must be all str or all sympy.Symbol.")

    # --- Parse every term ------------------------------------------------- #
    expr = Integer(0)
    i = 0
    while i < len(parts):
        # Read coefficient token
        coeff_str = parts[i]
        if not coeff_str.startswith("C"):
            raise ValueError(f"Expected 'C' token at position {i}, got {coeff_str}.")
        coeff = Integer(coeff_str[1:])
        i += 1

        # Read exponent tokens
        exps = []
        for _ in range(n_vars):
            if i >= len(parts) or not parts[i].startswith("E"):
                raise ValueError(f"Missing 'E' token at position {i}.")
            exps.append(Integer(parts[i][1:]))
            i += 1

        # Build term: coeff * Π v**e
        term = coeff
        for v, e in zip(vars_, exps):
            term *= v**e
        expr += term

    return expr


def parse_poly(text: str, var_names: Sequence[str | Symbol] | None = None) -> Expr:
    """Convert a math expression string or token sequence to a SymPy polynomial.

    This function handles:
    1. Standard mathematical notation (e.g., "4*x0 + 4*x1").
    2. SageMath-style power notation (e.g., "3*x0^2 + 3*x0").
    3. Internal token format (e.g., "C4 E1 E0 C4 E0 E1").

    Args:
        text (str):
            The mathematical expression or token sequence to parse.
        var_names (Sequence[str | sympy.Symbol] | None, optional):
            Variable names. Primarily used for the token sequence format to ensure
            the correct number of variables. For expression strings, variables are
            inferred, but providing them can ensure they are treated as symbols.

    Returns:
        sympy.Expr: A SymPy expression for the polynomial.
    """
    text = text.strip()

    # Heuristic: if the text starts with a 'C' token, attempt to parse it
    # using the token-based parser first.
    if text.startswith("C"):
        try:
            return _parse_poly_from_tokens(text, var_names)
        except (ValueError, IndexError):
            # Fallback to standard expression parsing if token parsing fails.
            # This allows parsing expressions that happen to start with 'C'
            # (e.g., if 'C' is a variable name).
            pass

    # Standard expression parsing
    # Replace SageMath-style power operator '^' with SymPy's '**'
    text_sympy = text.replace("^", "**")

    # Prepare a local dictionary of symbols if var_names are provided
    local_dict = {}
    if var_names:
        if all(isinstance(v, Symbol) for v in var_names):
            symbols_map = {s.name: s for s in var_names}
        else:
            symbols_map = {str(name): Symbol(str(name)) for name in var_names}
        local_dict.update(symbols_map)

    return parse_expr(text_sympy, local_dict=local_dict, evaluate=True)
