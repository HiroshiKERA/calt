"""Load preprocessor: convert polynomials from pickle to C/E expanded form.

Converts each polynomial to "C<coeff> E<e1> E<e2> ..." per term, joined by " + ".
Multiple polynomials (e.g. problem/solution as lists) are joined by " || ".

Works with both SymPy (PolyElement) and SageMath polynomials.
"""

from typing import Any

from ..load_preprocessor import DatasetLoadPreprocessor, _to_str


def _poly_terms(poly: Any) -> list[tuple[tuple[int, ...], Any]]:
    """Return list of (exponent_tuple, coefficient) for SymPy or SageMath polynomial."""
    # SymPy PolyElement: .terms() returns ((monom, coeff), ...)
    try:
        from sympy.polys.rings import PolyElement
        if isinstance(poly, PolyElement):
            return [(monom, coeff) for monom, coeff in poly.terms()]
    except ImportError:
        pass
    # SageMath polynomial: .dict() -> {exponent_tuple: coefficient}
    if hasattr(poly, "dict"):
        return [(exp, c) for exp, c in poly.dict().items() if c != 0]
    raise TypeError(
        f"Expected SymPy PolyElement or SageMath polynomial, got {type(poly).__name__}"
    )


def _coeff_to_int_or_str(c: Any) -> str:
    """Format coefficient for C token (integer or integer-like)."""
    try:
        n = int(c)
        return str(n)
    except (TypeError, ValueError):
        return str(c)


def poly_to_expanded_form(poly: Any) -> str:
    """Convert a single polynomial to expanded form: C<coeff> E<e1> E<e2> ... + ...

    Supports SymPy PolyElement and SageMath polynomial.
    """
    terms_list = _poly_terms(poly)
    if not terms_list:
        return "C0"

    parts = []
    for exp_tuple, coeff in terms_list:
        coeff_str = _coeff_to_int_or_str(coeff)
        term_tokens = [f"C{coeff_str}"] + [f"E{e}" for e in exp_tuple]
        parts.append(" ".join(term_tokens))
    return " + ".join(parts)


def obj_to_expanded_form(obj: Any, delimiter: str = " || ") -> str:
    """Convert problem/solution (single poly or list of polys) to expanded form string.

    - Single polynomial -> one "C... E... + ..." string.
    - List of polynomials -> each converted and joined by delimiter (default " || ").
    """
    if isinstance(obj, list):
        return delimiter.join(poly_to_expanded_form(p) for p in obj)
    return poly_to_expanded_form(obj)


class ExpandedFormLoadPreprocessor:
    """Convert pickle-loaded polynomials to C/E expanded form (input_text, target_text).

    Expects source to be a dict with "problem" and "solution" (as from pickle or JSONL
    that stored raw polynomial objects). Problem and solution can be a single polynomial
    or a list of polynomials. Each polynomial is converted to:
    "C<coeff> E<e1> E<e2> ... + C<coeff> E<e1> ..."
    Multiple polynomials are joined by delimiter (default " || ").
    """

    def __init__(self, delimiter: str = " || "):
        self.delimiter = delimiter

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError(
                "ExpandedFormLoadPreprocessor expects dict source (e.g. from pickle), "
                f"got {type(source).__name__}"
            )
        problem = source.get("problem")
        solution = source.get("solution")
        if problem is None or solution is None:
            raise ValueError("Source must have 'problem' and 'solution' keys")
        input_text = obj_to_expanded_form(problem, delimiter=self.delimiter)
        target_text = obj_to_expanded_form(solution, delimiter=self.delimiter)
        return input_text, target_text
