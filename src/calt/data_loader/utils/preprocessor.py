from typing import List, Any, Tuple
from abc import ABC, abstractmethod
import logging

# Basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class TermParseException(Exception):
    """Custom exception raised during term parsing errors."""

    pass


class AbstractPreprocessor(ABC):
    """Base abstract class for all preprocessors."""

    def __init__(self, num_variables: int, max_degree: int, max_coeff: int):
        """
        Initialize preprocessor parameters.

        Args:
            num_variables: Number of variables in the polynomial (e.g., x0, x1, ...)
            max_degree: Maximum degree of the polynomial
            max_coef: Maximum degree of any variable in the polynomial
        """
        if num_variables < 0:
            raise ValueError("num_variables must be positive")
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if max_coeff <= 0:
            raise ValueError("max_coef must be positive")

        self.num_variables = num_variables
        self.max_degree = max_degree
        self.max_coef = max_coeff
        self.var_name_to_index = {f"x{i}": i for i in range(num_variables)}

    def __call__(self, texts: List[str]) -> List[Any]:
        """Process texts (convenience wrapper for process method)."""
        return self.process(texts)

    @abstractmethod
    def process(self, texts: List[str]) -> List[Any]:
        """Abstract method for text processing to be implemented by subclasses."""
        raise NotImplementedError


class SymbolicToInternalProcessor(AbstractPreprocessor):
    """
    Convert symbolic mathematical expressions (SageMath-style) to internal token representation.

    Example:
        "2*x1^2*x0 + 5*x0 - 3" -> "C2 E1 E2 C5 E1 E0 C-3 E0 E0" (for num_vars=2)

    The internal representation uses:
        - 'C{n}' tokens for coefficients (e.g., C2, C-3)
        - 'E{n}' tokens for exponents (e.g., E1, E2, E0)
    Each term is represented as a coefficient token followed by exponent tokens for each variable.
    """

    def _log_warning(self, message: str, term_str: str) -> None:
        """Format and log a warning message about a term."""
        logging.warning(f"{message} in term '{term_str}'")

    def _get_zero_term(self) -> Tuple[int, List[int]]:
        """Return a representation of the zero term (coefficient 0, all exponents 0)."""
        return (0, [0] * self.num_variables)

    def _create_exponent_vector(self) -> List[int]:
        """Create a new exponent vector with all zeros."""
        return [0] * self.num_variables

    def _get_zero_exponents_str(self) -> str:
        """Generate string representation of zero exponents vector ("E0 E0 ...")."""
        return " ".join(["E0"] * self.num_variables)

    def _parse_term(self, term_str: str) -> Tuple[int, List[int]]:
        """Parse a term and return the coefficient and exponent vector.

        Args:
            term_str: String representation of a single term like "2*x0^2*x1"

        Returns:
            Tuple of (coefficient, exponent_vector)

        Raises:
            TermParseException: If the term cannot be parsed correctly
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

    def _format_internal(self, terms: List[Tuple[int, List[int]]]) -> str:
        """Convert parsed terms to the internal token representation string.

        Args:
            terms: List of (coefficient, exponent_vector) tuples

        Returns:
            String in the internal representation format
        """
        if not terms:
            return f"C0 {self._get_zero_exponents_str()}"

        internal_term_strs = []
        for coeff, exponents in terms:
            if coeff == 0:
                continue

            coeff_token = f"C{coeff}"
            if len(exponents) != self.num_variables:
                raise ValueError(
                    (
                        "Internal: Exp len mismatch "
                        f"(coeff {coeff}). Want {self.num_variables}, "
                        f"got {len(exponents)}."
                    )
                )
            exponent_tokens = [f"E{e}" for e in exponents]
            term_str = f"{coeff_token} {' '.join(exponent_tokens)}"
            internal_term_strs.append(term_str)

        if not internal_term_strs:
            return f"C0 {self._get_zero_exponents_str()}"

        return " ".join(internal_term_strs)

    def _poly_to_internal(self, poly_str: str) -> str:
        """Helper to convert a single polynomial string to internal representation.

        Args:
            poly_str: String representation of a polynomial

        Returns:
            String in the internal token format
        """
        tgt = poly_str.strip()
        if tgt == "" or tgt == "0":
            return f"C0 {self._get_zero_exponents_str()}"

        # Normalize: remove spaces, convert '-' to '+-' for easier splitting
        tgt = tgt.replace(" ", "")
        tgt = tgt.replace("-", "+-")
        if tgt.startswith("+"):
            tgt = tgt[1:]

        term_strs = [t.strip() for t in tgt.split("+") if t.strip()]

        parsed_terms: List[Tuple[int, List[int]]] = []
        for term_str in term_strs:
            try:
                coeff, exponents = self._parse_term(term_str)
                if coeff != 0:
                    parsed_terms.append((coeff, exponents))
            except Exception:
                return "[ERROR_PARSING]"

        return self._format_internal(parsed_terms)

    def process(self, text: str) -> str:
        """Process a symbolic text into internal token representation.

        If the text contains the '|' separator character, each part is processed
        separately and joined with '[SEP]' token.

        Args:
            text: Input symbolic text to process

        Returns:
            String in the internal token representation
        """
        # If text contains '|', process each part separately and join with [SEP]
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            internals = [self._poly_to_internal(p) for p in parts]
            processed_string = " [SEP] ".join(internals)
        else:
            processed_string = self._poly_to_internal(text)

        return processed_string


class IntegerToInternalProcessor(AbstractPreprocessor):
    """
    Convert an integer string, potentially containing '|' separators,
    to its internal token representation.

    Input format examples:
        - "12345"
        - "123|45|678"
    Output format examples:
        - "C1 C2 C3 C4 C5"
        - "C1 C2 C3 [SEP] C4 C5 [SEP] C6 C7 C8"

    The internal representation uses 'C{n}' tokens for digits.
    Parts separated by '|' are converted individually and joined by '[SEP]'.
    Note: num_variables, max_degree, max_coeff are inherited but not directly used.
    """

    def __init__(self, max_coeff: int = 9):
        """
        Initialize the processor.

        Args:
            max_coeff: The maximum digit value (typically 9).
                       Passed to superclass but primarily used for validation context.
        """
        # Use dummy values for num_variables and max_degree as they are not relevant
        super().__init__(num_variables=0, max_degree=0, max_coeff=max_coeff)

    def _number_to_tokens(self, number_str: str) -> str:
        """Convert a string of digits to space-separated 'C{digit}' tokens."""
        number_str = number_str.strip()  # Strip whitespace from individual parts
        if not number_str.isdigit():
            logging.warning(f"Invalid number format encountered: '{number_str}'")
            return "[ERROR_FORMAT]"
        return " ".join([f"C{digit}" for digit in number_str])

    def process(self, text: str) -> str:
        """Process an integer string (potentially with '|' separators)
        into internal token representation.

        Args:
            text: Input string representing one or more integers separated by '|'.

        Returns:
            String in the internal token representation (e.g., "C1 C2 [SEP] C3 C4"),
            or "[ERROR_FORMAT]" if any part is not a valid integer string.
        """
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            tokenized_parts = []
            for part in parts:
                tokens = self._number_to_tokens(part)
                if tokens == "[ERROR_FORMAT]":
                    # If any part fails, return error for the whole input
                    return "[ERROR_FORMAT]"
                tokenized_parts.append(tokens)
            # Join the tokenized parts with [SEP]
            return " [SEP] ".join(tokenized_parts)
        else:
            # If no separator, process the whole string
            return self._number_to_tokens(text.strip())
