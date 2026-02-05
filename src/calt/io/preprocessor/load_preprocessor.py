"""Dataset load-time preprocessors.

These run once at load time (before lexer) to convert raw file content
(text line, JSONL object, or pickle sample) into (input_text, target_text) pairs.
Users can provide their own implementations or use library-provided ones.
"""

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class DatasetLoadPreprocessor(Protocol):
    """Protocol for load-time preprocessors.

    Accepts one sample from file (a text line or a parsed JSONL object)
    and returns (input_text, target_text) for training.
    """

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        """Process one sample from file into (input_text, target_text).

        Args:
            source: For text files, a single line (str). For JSONL,
                a dict with "problem" and "solution" keys.

        Returns:
            (input_text, target_text) pair to feed into StandardDataset.
        """
        ...


class TextDefaultLoadPreprocessor:
    """Default preprocessor for text files: split by '#' into input and target."""

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, str):
            raise TypeError(
                f"TextDefaultLoadPreprocessor expects str, got {type(source).__name__}"
            )
        line = source.strip()
        if "#" not in line:
            raise ValueError(f"Text line has no '#' delimiter: {line[:80]!r}...")
        input_part, target_part = line.split("#", 1)
        return input_part.strip(), target_part.strip()


def _to_str(x: Any) -> str:
    """Convert problem/solution to string; handle nested lists via join.
    Exported for use by load_preprocessors (e.g. last_element)."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_to_str(item) for item in x)
    if isinstance(x, dict):
        return str(x)
    return str(x)


class JsonlDefaultLoadPreprocessor:
    """Default preprocessor for JSONL: stringify problem and solution."""

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError(
                f"JsonlDefaultLoadPreprocessor expects dict, got {type(source).__name__}"
            )
        problem = source.get("problem")
        solution = source.get("solution")
        if problem is None or solution is None:
            raise ValueError("JSONL object must have 'problem' and 'solution' keys")
        return _to_str(problem), _to_str(solution)


class UserCallableLoadPreprocessor:
    """Wraps a user callable (problem, solution) -> (input_text, target_text).

    For text sources, the line is split by '#' and the two parts are passed
    as (input_part, target_part). For JSONL sources, (problem, solution)
    from the parsed object are passed. The user can use SageMath etc.
    inside their callable.
    """

    def __init__(
        self,
        callable_: Callable[[Any, Any], tuple[str, str]],
        source_type: str = "auto",
    ):
        """Initialize with user callable.

        Args:
            callable_: (problem, solution) -> (input_text, target_text).
                problem/solution are str for text, or JSON types for JSONL.
            source_type: "text", "jsonl", or "auto". If "auto", infers from
                type of source in process_sample (str -> text, dict -> jsonl).
        """
        self.callable_ = callable_
        self.source_type = source_type

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if isinstance(source, str):
            if self.source_type == "jsonl":
                raise TypeError(
                    "UserCallableLoadPreprocessor configured for jsonl but got str"
                )
            # Text: split by # and pass (input_part, target_part)
            line = source.strip()
            if "#" not in line:
                raise ValueError(f"Text line has no '#' delimiter: {line[:80]!r}...")
            input_part, target_part = line.split("#", 1)
            return self.callable_(input_part.strip(), target_part.strip())

        if isinstance(source, dict):
            if self.source_type == "text":
                raise TypeError(
                    "UserCallableLoadPreprocessor configured for text but got dict"
                )
            problem = source.get("problem")
            solution = source.get("solution")
            if problem is None or solution is None:
                raise ValueError("JSONL object must have 'problem' and 'solution' keys")
            return self.callable_(problem, solution)

        raise TypeError(f"source must be str or dict, got {type(source).__name__}")


class PickleDefaultLoadPreprocessor:
    """Default preprocessor for pickle: problem/solution are Python objects (e.g. SageMath).
    Stringify them for training; user preprocessors can do math before stringifying.
    """

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError(
                "PickleDefaultLoadPreprocessor expects dict source (problem, solution)"
            )
        problem = source.get("problem")
        solution = source.get("solution")
        if problem is None or solution is None:
            raise ValueError("Source must have 'problem' and 'solution' keys")
        return _to_str(problem), _to_str(solution)
