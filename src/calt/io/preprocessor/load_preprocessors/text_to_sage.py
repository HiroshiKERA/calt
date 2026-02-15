"""Load preprocessor: parse text line into SageMath polynomial lists (dict).

Expects lines like "poly1 | poly2 | poly3 # poly4 | poly5 | poly6" (problem # solution).
Returns dict with "problem" and "solution" (lists of SageMath polynomials) so that
ExpandedFormLoadPreprocessor can be chained after it.
"""

from typing import Any, Callable


class TextToSageLoadPreprocessor:
    """Parse text line into SageMath polynomial lists (dict for chaining).

    Expects source to be a string line: "poly1 | poly2 # poly3 | poly4" (problem # solution).
    Splits by delimiter to get polynomial strings, parses each with ring(poly_str),
    returns {"problem": [poly, ...], "solution": [poly, ...]} for use with
    ExpandedFormLoadPreprocessor (e.g. via ChainLoadPreprocessor).

    Args:
        delimiter: Separator between polynomials in input text (default " | ").
        ring: Callable that takes a string and returns a polynomial (e.g. SageMath
              polynomial ring R so that R("9*x0 + 5*x2 + 10") works).
    """

    def __init__(
        self,
        delimiter: str = " | ",
        ring: Callable[[str], Any] | None = None,
    ):
        if ring is None:
            raise ValueError("TextToSageLoadPreprocessor requires ring")
        self.delimiter = delimiter
        self.ring = ring

    def _parse_part(self, part: str) -> list[Any]:
        """Split part by delimiter and parse each token with ring."""
        part = part.strip()
        if not part:
            return []
        tokens = [s.strip() for s in part.split(self.delimiter) if s.strip()]
        return [self.ring(t) for t in tokens]

    def process_sample(self, source: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(source, dict):
            raise TypeError(
                "TextToSageLoadPreprocessor expects str source (text line), got dict"
            )
        line = source.strip()
        if "#" not in line:
            raise ValueError(
                f"Text line must contain ' # ' separating problem and solution: {line[:80]!r}..."
            )
        problem_part, solution_part = line.split("#", 1)
        problem = self._parse_part(problem_part)
        solution = self._parse_part(solution_part)
        return {"problem": problem, "solution": solution}
