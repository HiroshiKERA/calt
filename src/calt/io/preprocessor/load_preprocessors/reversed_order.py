"""Reversed-order load preprocessor: target sequence is reversed by delimiter."""

from typing import Any

from ..load_preprocessor import _get_answer_from_source, _to_str


class ReversedOrderLoadPreprocessor:
    """Reverse the order of answer elements (split by delimiter, reverse, rejoin).

    - Text line: ``"11,4,11,4 # 11,15,9,13"`` → input: ``"11,4,11,4"``, target: ``"13,9,15,11"``
    - JSONL: same for ``{"problem": ..., "answer": ...}`` (or "solution"); split answer by delimiter, reverse, rejoin.
    """

    def __init__(self, problem_to_str: Any = None, delimiter: str = ","):
        self.problem_to_str = problem_to_str or _to_str
        self.delimiter = delimiter

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        # Text line ("11,4,11,4 # 11,15,9,13") case (format: problem # answer)
        if isinstance(source, str):
            line = source.strip()
            if "#" not in line:
                raise ValueError(
                    f"ReversedOrderLoadPreprocessor: expected '#' delimiter in text line, got: {line!r}"
                )
            problem_str, answer_str = line.split("#", 1)
            input_text = problem_str.strip()
            s = answer_str.strip()
            target_text = self._reverse_sequence(s)
            return input_text, target_text

        # JSONL / pickle dict form {"problem": ..., "answer": ...} (or "solution") case
        if not isinstance(source, dict):
            raise TypeError("ReversedOrderLoadPreprocessor expects str or dict source")

        problem = source.get("problem")
        answer = _get_answer_from_source(source)
        if problem is None or answer is None:
            raise ValueError(
                "Source must have 'problem' and 'answer' (or 'solution') keys"
            )

        input_text = self.problem_to_str(problem)
        target_text = self._answer_to_reversed_str(answer)
        return input_text, target_text

    def _reverse_sequence(self, s: str) -> str:
        """Split by delimiter, reverse, and rejoin."""
        if not s.strip():
            return s
        if self.delimiter not in s:
            return s
        tokens = [tok.strip() for tok in s.split(self.delimiter) if tok.strip()]
        return self.delimiter.join(reversed(tokens))

    def _answer_to_reversed_str(self, answer: Any) -> str:
        """Return answer (list or str) as a reversed string."""
        if isinstance(answer, list) and answer:
            parts = [
                _to_str(x) if not isinstance(x, str) else x for x in reversed(answer)
            ]
            return self.delimiter.join(parts)
        if isinstance(answer, str):
            return self._reverse_sequence(answer)
        return _to_str(answer)
