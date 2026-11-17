"""Load preprocessor: chain multiple load preprocessors in sequence."""

from typing import Any


class ChainLoadPreprocessor:
    """Run multiple load preprocessors in sequence.

    The first receives the raw source (str or dict); each next receives the
    previous output. The last preprocessor must return (input_text, target_text),
    optionally followed by a dict of extra labels.
    Use this to combine e.g. TextToSageLoadPreprocessor and ExpandedFormLoadPreprocessor.
    """

    def __init__(self, *preprocessors: Any):
        if not preprocessors:
            raise ValueError("ChainLoadPreprocessor requires at least one preprocessor")
        self.preprocessors = list(preprocessors)

    def process_sample(self, source: str | dict[str, Any]) -> tuple:
        x: Any = source
        for p in self.preprocessors:
            x = p.process_sample(x)
        if not isinstance(x, tuple) or len(x) not in (2, 3):
            raise TypeError(
                "Last preprocessor must return (input_text, target_text) or "
                f"(input_text, target_text, extra_labels), got {type(x)}"
            )
        return x
