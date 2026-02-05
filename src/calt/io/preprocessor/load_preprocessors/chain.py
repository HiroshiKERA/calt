"""Load preprocessor: chain multiple load preprocessors in sequence."""

from typing import Any


class ChainLoadPreprocessor:
    """Run multiple load preprocessors in sequence.

    The first receives the raw source (str or dict); each next receives the
    previous output. The last preprocessor must return (input_text, target_text).
    Use this to combine e.g. TextToSageLoadPreprocessor and ExpandedFormLoadPreprocessor.
    """

    def __init__(self, *preprocessors: Any):
        if not preprocessors:
            raise ValueError("ChainLoadPreprocessor requires at least one preprocessor")
        self.preprocessors = list(preprocessors)

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        x: Any = source
        for p in self.preprocessors:
            x = p.process_sample(x)
        if not isinstance(x, tuple) or len(x) != 2:
            raise TypeError(
                f"Last preprocessor must return (input_text, target_text), got {type(x)}"
            )
        return x[0], x[1]
