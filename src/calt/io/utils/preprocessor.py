import logging
import warnings
from abc import ABC, abstractmethod
from typing import Iterable

# Basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class TermParseException(Exception):
    """Custom exception raised during term parsing errors."""

    pass


class AbstractPreprocessor(ABC):
    """Base abstract class for all preprocessors."""

    def __init__(self, num_variables: int, max_degree: int, max_coeff: int):
        """Initialize preprocessor parameters.

        Args:
            num_variables (int): Number of variables in the polynomial (e.g., x0, x1, ...).
            max_degree (int): Maximum degree of the polynomial.
            max_coeff (int): Maximum coefficient value in the polynomial.
        """
        if num_variables < 0:
            raise ValueError("num_variables must be positive")
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if max_coeff <= 0:
            raise ValueError("max_coeff must be positive")

        self.num_variables = num_variables
        self.max_degree = max_degree
        self.max_coeff = max_coeff
        self.var_name_to_index = {f"x{i}": i for i in range(num_variables)}

    def __call__(self, text: str) -> str:
        """Process text (convenience wrapper for process method)."""
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> str:
        """Abstract method for text processing to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: str) -> str:
        """Abstract method for token processing to be implemented by subclasses."""
        raise NotImplementedError

    # For backward compatibility: process is an alias for to_internal
    def process(self, text: str) -> str:
        return self.encode(text)


class ProcessorChain(AbstractPreprocessor):
    """Compose multiple preprocessors and apply them sequentially."""

    def __init__(self, processors: Iterable["AbstractPreprocessor"]) -> None:
        processors = list(processors)
        if not processors:
            raise ValueError("ProcessorChain requires at least one preprocessor.")

        first = processors[0]
        super().__init__(
            num_variables=first.num_variables,
            max_degree=first.max_degree,
            max_coeff=first.max_coeff,
        )
        self.processors = processors

    def encode(self, text: str) -> str:
        for processor in self.processors:
            text = processor.encode(text)
        return text

    def decode(self, tokens: str) -> str:
        for processor in reversed(self.processors):
            tokens = processor.decode(tokens)
        return tokens


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


class PolynomialToInternalProcessor(AbstractPreprocessor):
    """Convert SageMath-style expressions to/from internal token representation.

    Example (to_internal):
        "2*x1^2*x0 + 5*x0 - 3" -> "C2 E1 E2 C5 E1 E0 C-3 E0 E0" (for ``num_vars=2``)

    Example (to_original):
        "C2 E2 E1 C5 E1 E0 C-3 E0 E0" -> "2*x0^2*x1 + 5*x0 - 3"

    The internal representation uses:
        - ``C{n}`` tokens for coefficients (e.g., ``C2``, ``C-3``)
        - ``E{n}`` tokens for exponents (e.g., ``E1``, ``E2``, ``E0``)
    Each term is represented as a coefficient token followed by exponent tokens for each variable.
    """

    def __init__(
        self,
        num_variables: int,
        max_degree: int,
        max_coeff: int,
        digit_group_size: int | None = None,
    ):
        super().__init__(
            num_variables=num_variables, max_degree=max_degree, max_coeff=max_coeff
        )
        self.digit_group_size = digit_group_size

    def _log_warning(self, message: str, term_str: str) -> None:
        """Format and log a warning message about a term."""
        logging.warning(f"{message} in term '{term_str}'")

    def _get_zero_term(self) -> tuple[int, list[int]]:
        """Return the zero term (coefficient 0, all exponents 0)."""
        return (0, [0] * self.num_variables)

    def _create_exponent_vector(self) -> list[int]:
        """Create a new exponent vector with all zeros."""
        return [0] * self.num_variables

    def _get_zero_exponents_str(self) -> str:
        """Generate string for zero exponents vector ("E0 E0 ...")."""
        return " ".join(["E0"] * self.num_variables)

    def _parse_term(self, term_str: str) -> tuple[int, list[int]]:
        """Parse a term and return the coefficient and exponent vector.

        Args:
            term_str (str): String representation of a single term like "2*x0^2*x1".

        Returns:
            tuple[int, list[int]]: Pair of (coefficient, exponent_vector).

        Raises:
            TermParseException: If the term cannot be parsed correctly.
        """
        term_str = term_str.strip()
        if not term_str:
            return self._get_zero_term()

        exponents = self._create_exponent_vector()
        coeff = 1
        sign = 1

        if term_str.startswith("-"):
            sign = -1
            term_str = term_str[1:].strip()
        elif term_str.startswith("+"):
            term_str = term_str[1:].strip()

        parts = [p.strip() for p in term_str.split("*")]
        coeff_part_found = False
        processed_parts = []

        if parts[0].isdigit():
            coeff = int(parts[0])
            coeff_part_found = True
            processed_parts = parts[1:]
        else:
            processed_parts = parts

        variable_parts_exist = False
        for part in processed_parts:
            if not part:
                continue

            var_name = part
            exponent = 1

            if "^" in part:
                base, exp_str = part.split("^", 1)
                var_name = base.strip()
                exp_str = exp_str.strip()
                if not exp_str.isdigit():
                    raise TermParseException(
                        f"Invalid exponent '{exp_str}' in term '{term_str}'"
                    )
                exponent = int(exp_str)

            if var_name in self.var_name_to_index:
                var_index = self.var_name_to_index[var_name]
                exponents[var_index] = exponent
                variable_parts_exist = True
            elif var_name.isdigit() and not coeff_part_found:
                coeff = int(var_name)
                coeff_part_found = True
            else:
                raise TermParseException(
                    f"Unknown var/part '{var_name}' in term '{term_str}'"
                )

        final_coeff = sign * coeff

        # For constant terms (no variables)
        if not variable_parts_exist and coeff_part_found:
            return (final_coeff, self._create_exponent_vector())

        # For variable terms without explicit coefficient
        if not variable_parts_exist and not coeff_part_found:
            if term_str in self.var_name_to_index:
                var_index = self.var_name_to_index[term_str]
                exponents[var_index] = 1
                return (sign * 1, exponents)
            elif term_str == "1":
                return (sign * 1, self._create_exponent_vector())
            else:
                raise TermParseException(f"Cannot parse term '{term_str}'")

        if variable_parts_exist and not coeff_part_found:
            return (sign * 1, exponents)

        return (final_coeff, exponents)

    def _chunk_numeric_string(self, digits: str, k: int) -> list[str]:
        """Split a numeric string into groups of size ``k`` without zero padding."""
        if k <= 0 or not digits:
            return [digits]

        chunks: list[str] = []
        idx = len(digits)
        while idx > 0:
            start = max(0, idx - k)
            chunks.append(digits[start:idx])
            idx = start
        chunks.reverse()
        return chunks

    def _number_to_tokens(self, number_str: str, k: int | None) -> list[str]:
        """Return coefficient tokens for a signed integer string.

        Applies digit grouping only when ``k`` is positive; otherwise keeps the entire
        literal (including any leading zeros) as a single token.
        """
        stripped = number_str.strip()
        if not stripped:
            raise ValueError(f"Invalid integer literal '{number_str}'")

        sign = ""
        digits = stripped
        if digits[0] in "+-":
            if digits[0] == "-":
                sign = "-"
            digits = digits[1:]
        if not digits or not digits.isdigit():
            raise ValueError(f"Invalid integer literal '{number_str}'")

        if not k or k <= 0:
            payload = digits if not sign else f"-{digits}"
            return [f"C{payload}"]

        chunks = self._chunk_numeric_string(digits, k)
        if sign == "-":
            chunks[0] = f"-{chunks[0]}"
        return [f"C{chunk}" for chunk in chunks]

    def _format_internal(self, terms: list[tuple[int, list[int]]]) -> str:
        """Convert parsed terms to the internal token representation string.

        Args:
            terms (list[tuple[int, list[int]]]): List of (coefficient, exponent_vector) tuples.

        Returns:
            str: String in the internal representation format.
        """
        if not terms:
            zero_tokens = ["C0"]
            if self.num_variables:
                zero_tokens.extend(["E0"] * self.num_variables)
            return " ".join(zero_tokens)

        internal_term_strs: list[str] = []
        for coeff, exponents in terms:
            if coeff == 0:
                continue

            if len(exponents) != self.num_variables:
                raise ValueError(
                    (
                        "Internal: Exp len mismatch "
                        f"(coeff {coeff}). Want {self.num_variables}, "
                        f"got {len(exponents)}."
                    )
                )
            coeff_tokens = self._number_to_tokens(str(coeff), self.digit_group_size)
            exponent_tokens = [f"E{e}" for e in exponents]
            term_tokens = coeff_tokens + exponent_tokens
            internal_term_strs.append(" ".join(term_tokens))

        if not internal_term_strs:
            zero_tokens = ["C0"]
            if self.num_variables:
                zero_tokens.extend(["E0"] * self.num_variables)
            return " ".join(zero_tokens)

        return " ".join(internal_term_strs)

    def _poly_to_encode(self, poly_str: str) -> str:
        """Convert a single polynomial string to internal representation.

        Args:
            poly_str (str): String representation of a polynomial.

        Returns:
            str: String in the internal token format.
        """
        tgt = poly_str.strip()
        if tgt == "" or tgt == "0":
            zero_tokens = ["C0"]
            if self.num_variables:
                zero_tokens.extend(["E0"] * self.num_variables)
            return " ".join(zero_tokens)

        # Normalize: remove spaces, convert '-' to '+-' for easier splitting
        tgt = tgt.replace(" ", "")
        tgt = tgt.replace("-", "+-")
        if tgt.startswith("+"):
            tgt = tgt[1:]

        term_strs = [t.strip() for t in tgt.split("+") if t.strip()]

        parsed_terms: list[tuple[int, list[int]]] = []
        for term_str in term_strs:
            try:
                coeff, exponents = self._parse_term(term_str)
                if coeff != 0:
                    parsed_terms.append((coeff, exponents))
            except Exception:
                return "[ERROR_PARSING]"

        return self._format_internal(parsed_terms)

    def encode(self, text: str) -> str:
        """Process a symbolic text into internal token representation.

        If the text contains the '|' separator character, each part is processed
        separately and joined with '[SEP]' token.

        Args:
            text (str): Input symbolic text to process.

        Returns:
            str: String in the internal token representation.
        """
        if self.num_variables == 0:
            stripped_text = text.strip()
            if stripped_text == "":
                return "[ERROR_FORMAT]"
            if "|" in stripped_text:
                parts = [p.strip() for p in stripped_text.split("|")]
                encoded_parts: list[str] = []
                for part in parts:
                    if not part:
                        logging.warning(f"Invalid number format encountered: '{part}'")
                        return "[ERROR_FORMAT]"
                    try:
                        tokens = self._number_to_tokens(part, self.digit_group_size)
                    except ValueError:
                        logging.warning(f"Invalid number format encountered: '{part}'")
                        return "[ERROR_FORMAT]"
                    encoded_parts.append(" ".join(tokens))
                return " [SEP] ".join(encoded_parts)
            try:
                tokens = self._number_to_tokens(stripped_text, self.digit_group_size)
            except ValueError:
                logging.warning(f"Invalid number format encountered: '{stripped_text}'")
                return "[ERROR_FORMAT]"
            return " ".join(tokens) if tokens else "[ERROR_FORMAT]"

        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            internals = [self._poly_to_encode(p) for p in parts]
            return " [SEP] ".join(internals)
        return self._poly_to_encode(text)

    def _internal_to_poly(self, tokens: str) -> str:
        """Convert a single internal token string to a polynomial."""
        stripped = tokens.strip()
        if stripped == "[ERROR_PARSING]":
            return "[ERROR_PARSING]"

        parts = stripped.split()
        if not parts or (len(parts) == self.num_variables + 1 and parts[0] == "C0"):
            return "0"

        terms = []
        i = 0
        while i < len(parts):
            coeff_tokens: list[str] = []
            while i < len(parts) and parts[i].startswith("C"):
                coeff_tokens.append(parts[i])
                i += 1

            if not coeff_tokens:
                logging.warning(
                    f"Invalid token sequence starting at index {i}: {parts[i:]}"
                )
                break

            exponent_tokens = parts[i : i + self.num_variables]
            if len(exponent_tokens) != self.num_variables or any(
                not token.startswith("E") for token in exponent_tokens
            ):
                logging.warning(
                    f"Invalid exponent sequence for coeff tokens {coeff_tokens}: {exponent_tokens}"
                )
                break
            i += self.num_variables

            if len(coeff_tokens) == 1 and coeff_tokens[0] == "C0":
                continue

            coeff_str_parts = [token[1:] for token in coeff_tokens]
            coeff_str = "".join(coeff_str_parts)
            coeff = int(coeff_str)
            if coeff == 0:
                continue

            exponents = [int(token[1:]) for token in exponent_tokens]

            term_str = ""
            if abs(coeff) == 1 and any(e > 0 for e in exponents):
                if coeff == -1:
                    term_str += "-"
            else:
                term_str += str(coeff)

            var_term_parts = []
            for var_idx, exp in enumerate(exponents):
                if exp > 0:
                    var_str = f"x{var_idx}"
                    if exp > 1:
                        var_str += f"^{exp}"
                    var_term_parts.append(var_str)

            if var_term_parts:
                if term_str and term_str != "-":
                    term_str += "*"
                term_str += "*".join(var_term_parts)

            terms.append(term_str)

        if not terms:
            return "0"

        # Join terms with signs
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result.replace(" ", "").replace("+-", "-")

    def decode(self, tokens: str) -> str:
        """Convert an internal token string back to a symbolic polynomial expression."""
        if self.num_variables == 0:
            stripped = tokens.strip()
            if stripped == "[ERROR_FORMAT]":
                return "[ERROR_FORMAT]"
            if "[SEP]" in stripped:
                parts = [p.strip() for p in stripped.split("[SEP]")]
            else:
                parts = [stripped]

            numbers: list[str] = []
            for part in parts:
                if not part:
                    numbers.append("")
                    continue

                tokens_in_part = part.split()
                sign = ""
                digits: list[str] = []
                for idx, token in enumerate(tokens_in_part):
                    if not token.startswith("C"):
                        logging.warning(f"Invalid integer token encountered: '{token}'")
                        return "[ERROR_FORMAT]"

                    payload = token[1:]
                    if idx == 0 and payload.startswith("-"):
                        sign = "-"
                        payload = payload[1:]
                    if not payload or not payload.isdigit():
                        logging.warning(f"Invalid integer token encountered: '{token}'")
                        return "[ERROR_FORMAT]"
                    digits.append(payload)
                numbers.append(f"{sign}{''.join(digits)}")
            return " | ".join(numbers)

        if "[SEP]" in tokens:
            parts = tokens.split("[SEP]")
            original_parts = [self._internal_to_poly(p.strip()) for p in parts]
            return " | ".join(original_parts)
        return self._internal_to_poly(tokens)


class IntegerToInternalProcessor(AbstractPreprocessor):
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
        super().__init__(num_variables=0, max_degree=0, max_coeff=max_coeff)
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
