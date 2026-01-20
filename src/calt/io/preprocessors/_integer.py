"""Integer to internal representation processor (deprecated)."""

import warnings

from .base import AbstractPostProcessor
from ._polynomial import PolynomialToInternalProcessor


class IntegerToInternalProcessor(AbstractPostProcessor):
    """Deprecated wrapper around :class:`PolynomialToInternalProcessor` for integer strings."""

    def __init__(
        self,
        max_coeff: int = 9,
        digit_group_size: int | None = None,
    ):
        warnings.warn(
            (
                "IntegerToInternalProcessor is deprecated; "
                "use PolynomialToInternalProcessor(num_variables=0, ...) instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        # Note: This class is deprecated and uses dummy values for AbstractPostProcessor
        # since it doesn't need num_variables, max_degree, max_coeff anymore
        super().__init__()
        self._delegate = PolynomialToInternalProcessor(
            num_variables=0,
            max_degree=0,
            max_coeff=max_coeff,
            digit_group_size=digit_group_size,
        )

    def encode(self, text: str) -> str:
        return self._delegate.encode(text)

    def decode(self, tokens: str) -> str:
        return self._delegate.decode(tokens)
