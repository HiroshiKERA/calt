"""Last-element load preprocessor for cumulative-sum style tasks."""

from typing import Any

from ..load_preprocessor import DatasetLoadPreprocessor, _to_str


class LastElementLoadPreprocessor:
    """Use only the last element of solution (e.g. cumulative-sum final value).

    Expects dict source with solution as a list (e.g. [92, 93, 98]);
    returns (input_text, str(last_element)) so the model predicts only the final value.
    """

    def __init__(self, problem_to_str: Any = None):
        self.problem_to_str = problem_to_str or _to_str

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError("LastElementLoadPreprocessor expects dict source")
        problem = source.get("problem")
        solution = source.get("solution")
        if problem is None or solution is None:
            raise ValueError("Source must have 'problem' and 'solution' keys")
        input_text = self.problem_to_str(problem)
        if isinstance(solution, list) and solution:
            last = solution[-1]
            target_text = _to_str(last) if not isinstance(last, str) else last
        else:
            target_text = _to_str(solution)
        return input_text, target_text
