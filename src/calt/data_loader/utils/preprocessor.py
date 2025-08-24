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
                    raise TermParseException(f"Invalid exponent '{exp_str}' in term '{term_str}'")
                exponent = int(exp_str)

            if var_name in self.var_name_to_index:
                var_index = self.var_name_to_index[var_name]
                exponents[var_index] = exponent
                variable_parts_exist = True
            elif var_name.isdigit() and not coeff_part_found:
                coeff = int(var_name)
                coeff_part_found = True
            else:
                raise TermParseException(f"Unknown var/part '{var_name}' in term '{term_str}'")

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

    def _format_internal(self, terms: list[tuple[int, list[int]]]) -> str:
        """Convert parsed terms to the internal token representation string.

        Args:
            terms (list[tuple[int, list[int]]]): List of (coefficient, exponent_vector) tuples.

        Returns:
            str: String in the internal representation format.
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

    def _poly_to_encode(self, poly_str: str) -> str:
        """Convert a single polynomial string to internal representation.

        Args:
            poly_str (str): String representation of a polynomial.

        Returns:
            str: String in the internal token format.
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
        # If text contains '|', process each part separately and join with [SEP]
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            internals = [self._poly_to_encode(p) for p in parts]
            processed_string = " [SEP] ".join(internals)
        else:
            processed_string = self._poly_to_encode(text)

        return processed_string

    def _internal_to_poly(self, tokens: str) -> str:
        """Convert a single internal token string to a polynomial."""
        parts = tokens.strip().split()
        if not parts or (len(parts) == self.num_variables + 1 and parts[0] == "C0"):
            return "0"

        terms = []
        i = 0
        while i < len(parts):
            # Each term consists of one C token and num_variables E tokens.
            term_parts = parts[i : i + self.num_variables + 1]
            i += self.num_variables + 1

            if not term_parts or not term_parts[0].startswith("C"):
                logging.warning(f"Invalid token sequence: {term_parts}")
                continue  # or raise error

            coeff_str = term_parts[0][1:]
            coeff = int(coeff_str)

            if coeff == 0:
                continue

            exponents = [int(p[1:]) for p in term_parts[1:]]

            term_str = ""
            # Coefficient part
            if abs(coeff) == 1 and any(e > 0 for e in exponents):
                if coeff == -1:
                    term_str += "-"
            else:
                term_str += str(coeff)

            # Variable parts
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
        if "[SEP]" in tokens:
            parts = tokens.split("[SEP]")
            original_parts = [self._internal_to_poly(p.strip()) for p in parts]
            return " | ".join(original_parts)
        else:
            return self._internal_to_poly(tokens)


class IntegerToInternalProcessor(AbstractPreprocessor):
    """Convert an integer string to/from internal token representation.

    Input format examples (to_internal):
        - "12345"
        - "123|45|678"
    Output format examples (from_internal):
        - "C1 C2 C3 C4 C5"
        - "C1 C2 C3 [SEP] C4 C5 [SEP] C6 C7 C8"

    The internal representation uses ``C{n}`` tokens for digits.
    Parts separated by '|' are converted individually and joined by ``[SEP]``.
    Note: ``num_variables``, ``max_degree``, ``max_coeff`` are inherited but not directly used.
    """

    def __init__(self, max_coeff: int = 9):
        """Initialize the processor.

        Args:
            max_coeff (int): The maximum digit value (typically 9). Passed to superclass but primarily used for validation context.
        """
        # Use dummy values for num_variables and max_degree as they are not relevant
        super().__init__(num_variables=0, max_degree=0, max_coeff=max_coeff)

    def _number_to_tokens(self, number_str: str) -> str:
        """Convert a string of digits to space-separated ``C{digit}`` tokens."""
        number_str = number_str.strip()  # Strip whitespace from individual parts
        if not number_str.isdigit():
            logging.warning(f"Invalid number format encountered: '{number_str}'")
            return "[ERROR_FORMAT]"
        return " ".join([f"C{digit}" for digit in number_str])

    def encode(self, text: str) -> str:
        """Process an integer string into internal token representation.

        Args:
            text (str): Input string representing one or more integers separated by '|'.

        Returns:
            str: Internal token representation (e.g., "C1 C2 [SEP] C3 C4"), or "[ERROR_FORMAT]" if any part is not a valid integer string.
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

    def _tokens_to_number(self, tokens_str: str) -> str:
        """Convert a string of ``C{digit}`` tokens back to a number string."""
        tokens_str = tokens_str.strip()
        if not tokens_str:
            return ""
        parts = tokens_str.split()
        return "".join([part[1:] for part in parts if part.startswith("C") and part[1:].isdigit()])

    def decode(self, tokens: str) -> str:
        """Convert an internal token representation back to an integer string."""
        if "[SEP]" in tokens:
            parts = tokens.split("[SEP]")
            # Process each part and join with '|'
            number_parts = [self._tokens_to_number(p.strip()) for p in parts]
            return "|".join(number_parts)
        else:
            # Process the whole string if no separator
            return self._tokens_to_number(tokens.strip())
