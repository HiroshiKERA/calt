"""Coefficient postfix processor for reordering coefficient and exponent tokens."""

from .base import AbstractPreprocessor


class CoefficientPostfixProcessor(AbstractPreprocessor):
    """Move coefficient tokens so they follow their exponent tokens within each term."""

    def __init__(
        self,
        coefficient_prefix: str = "C",
        exponent_prefix: str = "E",
    ) -> None:
        super().__init__(num_variables=0, max_degree=0, max_coeff=1)
        self.coefficient_prefix = coefficient_prefix
        self.exponent_prefix = exponent_prefix

    def encode(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        tokens = stripped.split()
        return " ".join(self._reorder_tokens(tokens, coefficients_last=True))

    def decode(self, tokens: str) -> str:
        stripped = tokens.strip()
        if not stripped:
            return tokens
        pieces = stripped.split()
        return " ".join(self._reorder_tokens(pieces, coefficients_last=False))

    def _reorder_tokens(
        self, tokens: list[str], *, coefficients_last: bool
    ) -> list[str]:
        reordered = []
        current_coeffs = []
        current_exponents = []

        def flush_term() -> None:
            nonlocal current_coeffs, current_exponents
            if not current_coeffs and not current_exponents:
                return
            if coefficients_last:
                reordered.extend(current_exponents)
                reordered.extend(current_coeffs)
            else:
                reordered.extend(current_coeffs)
                reordered.extend(current_exponents)
            current_coeffs = []
            current_exponents = []

        for token in tokens:
            is_coeff = token.startswith(self.coefficient_prefix)
            is_exp = token.startswith(self.exponent_prefix)

            if not (is_coeff or is_exp):
                flush_term()
                reordered.append(token)
                continue

            if coefficients_last:
                if is_coeff:
                    if current_exponents:
                        flush_term()
                    current_coeffs.append(token)
                else:
                    current_exponents.append(token)
            else:
                if is_exp:
                    if current_coeffs and current_exponents:
                        flush_term()
                    current_exponents.append(token)
                else:
                    current_coeffs.append(token)

        flush_term()
        return reordered
