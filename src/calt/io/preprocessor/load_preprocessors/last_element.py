"""Last-element load preprocessor for cumulative-sum style tasks."""

from typing import Any

from ..load_preprocessor import _get_answer_from_source, _to_str


class LastElementLoadPreprocessor:
    """Use only the last element of answer (e.g. cumulative-sum final value).

    - Text line: single line like ``"11,4,11,4 # 11,15,9,13"`` (format: problem # answer)
    - JSONL: dict with ``{"problem": ..., "answer": ...}`` (or "solution")
    - ``answer`` is one of:
      - list (e.g. ``[11, 15, 9, 13]``)
      - delimiter-joined string (e.g. ``"11,15,9,13"``)
    - Output is ``(input_text, last_answer_str)``; only the last element is used as target.
      e.g. ``"11,4,11,4 # 11,15,9,13"`` → input: ``"11,4,11,4"``, target: ``"13"``
    """

    def __init__(self, problem_to_str: Any = None, delimiter: str = ","):
        # Problem formatting is delegated to existing _to_str
        self.problem_to_str = problem_to_str or _to_str
        self.delimiter = delimiter

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        # Text line ("11,4,11,4 # 11,15,9,13") case (format: problem # answer)
        if isinstance(source, str):
            line = source.strip()
            if "#" not in line:
                raise ValueError(
                    f"LastElementLoadPreprocessor: expected '#' delimiter in text line, got: {line!r}"
                )
            problem_str, answer_str = line.split("#", 1)
            input_text = problem_str.strip()
            s = answer_str.strip()
            if self.delimiter in s:
                tokens = [tok.strip() for tok in s.split(self.delimiter) if tok.strip()]
                last = tokens[-1] if tokens else s
            else:
                last = s
            target_text = last
            return input_text, target_text

        # JSONL / pickle dict form {"problem": ..., "answer": ...} (or "solution") case
        if not isinstance(source, dict):
            raise TypeError("LastElementLoadPreprocessor expects str or dict source")

        problem = source.get("problem")
        answer = _get_answer_from_source(source)
        if problem is None or answer is None:
            raise ValueError(
                "Source must have 'problem' and 'answer' (or 'solution') keys"
            )

        # Input text as-is (or formatted via _to_str)
        input_text = self.problem_to_str(problem)

        # If answer is a list: use its last element
        if isinstance(answer, list) and answer:
            last = answer[-1]
        # If answer is delimiter-joined string: split and take last token
        elif isinstance(answer, str):
            s = answer.strip()
            if self.delimiter in s:
                tokens = [tok.strip() for tok in s.split(self.delimiter) if tok.strip()]
                last = tokens[-1] if tokens else s
            else:
                last = s
        else:
            # Otherwise treat as a single value
            last = answer

        target_text = _to_str(last) if not isinstance(last, str) else last
        return input_text, target_text
