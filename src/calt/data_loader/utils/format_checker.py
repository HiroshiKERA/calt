import logging
import re

from sage.all import QQ, RR, ZZ, PolynomialRing

# Set up logger for this module
logger = logging.getLogger(__name__)


class FormatChecker:
    """Format checker for SageMath dataset validation.

    This class validates that generated datasets are in the correct SageMath format
    before they are used as input to models. It ensures that all problems and solutions
    can be properly parsed and cast to SageMath's mathematical structures.

    The checker is designed to catch format errors early in the data pipeline,
    preventing issues when the dataset is loaded for model training or evaluation.
    """

    def __init__(
        self,
        variable_names: str | None = None,
        num_vars: int | None = None,
        variable_name: str | None = None,
    ):
        """Initialize the format checker.

        Args:
            variable_names (str | None): Comma-separated string of variable names (e.g., "x, y, z").
            num_vars (int | None): Number of variables to generate (required if ``variable_names`` is not specified).
            variable_name (str | None): Base name for variables when using ``num_vars`` (required if ``num_vars`` is specified).

        Raises:
            ValueError: If both ``variable_names`` and ``num_vars`` are specified, or if ``num_vars`` is specified without ``variable_name``.
        """
        self._validate_parameters(variable_names, num_vars, variable_name)
        self.variables = self._setup_variables(variable_names, num_vars, variable_name)
        self.polynomial_rings = self._setup_polynomial_rings()

    def _validate_parameters(self, variable_names, num_vars, variable_name):
        """Validate initialization parameters."""
        # Validate parameter combinations
        if variable_names is not None:
            if num_vars is not None:
                raise ValueError("Cannot specify both 'variable_names' and 'num_vars'")
            if variable_name is not None:
                raise ValueError(
                    "Cannot specify both 'variable_names' and 'variable_name'"
                )

        if num_vars is not None:
            if variable_name is None:
                raise ValueError("'num_vars' requires 'variable_name' to be specified")
            if num_vars <= 0:
                raise ValueError("'num_vars' must be positive")

    def _setup_variables(self, variable_names, num_vars, variable_name):
        """Set up variables based on parameters."""
        if variable_names is not None:
            # Parse comma-separated variable names into a list
            variables = [
                var.strip() for var in variable_names.split(",") if var.strip()
            ]
            if not variables:
                raise ValueError("No valid variables found in 'variable_names'")

            # Check for duplicate variable names
            if len(variables) != len(set(variables)):
                duplicates = [var for var in set(variables) if variables.count(var) > 1]
                raise ValueError(f"Duplicate variable names found: {duplicates}")
            return variables
        elif num_vars is not None:
            return [f"{variable_name}{i}" for i in range(num_vars)]
        return None

    def _setup_polynomial_rings(self):
        """Set up polynomial rings for efficiency."""
        if not self.variables:
            return []

        # Pre-generate polynomial rings for efficiency
        rings = []
        for ring in [RR, QQ, ZZ]:
            try:
                R = PolynomialRing(ring, self.variables)
                rings.append(R)
            except Exception as e:
                logger.warning(f"Failed to create PolynomialRing with {ring}: {e}")
        return rings

    def check_format(self, dataset: list[str], num_samples: int | None = None) -> bool:
        """Check if the dataset follows the correct SageMath format.

        This method validates that all problems and solutions in the dataset can be
        properly parsed and cast to SageMath's mathematical structures (PolynomialRing, RR, QQ, ZZ).
        It is designed to be run before using the dataset for model training or evaluation
        to ensure data quality and prevent runtime errors.

        Args:
            dataset (list[str]): List of dataset lines.
            num_samples (int | None, optional): Maximum number of samples to check. If ``None``, checks all samples.

        Returns:
            bool: ``True`` if format is valid, ``False`` otherwise.

        Example:
            >>> # variable_names is specified
            >>> checker = FormatChecker(variable_names="x, y, z")
            >>> dataset_lines = ["x^2 + y^2 # x^2 + y^2", "x + y | z + 1 # x + y | z + 1"]
            >>> checker.check_format(dataset_lines)
            True
            >>> # num_vars and variable_name are specified
            >>> checker = FormatChecker(num_vars=3, variable_name="x")
            >>> checker.check_format(dataset_lines)
            True
            >>> # Check only first 10 samples
            >>> checker = FormatChecker(num_vars=3, variable_name="x")
            >>> checker.check_format(dataset_lines, num_samples=10)
            True

            Example dataset format:
            # Example of polynomial samples
            x^2 + y^2 # x^2 + y^2
            x + y | z + 1 # x + y | z + 1
            x^3 + 2*x*y || y^2 + z # x^3 + 2*x*y || y^2 + z

            # Example of arithmetic samples
            2 + 3 # 5
            12 # 2 | 2 | 3
            217 # 7 | 31
        """
        if not isinstance(dataset, list):
            logger.error("dataset must be a list of strings")
            return False

        if num_samples is not None and num_samples <= 0:
            logger.error("num_samples must be positive or None")
            return False

        if num_samples is not None and num_samples > len(dataset):
            logger.warning(
                f"num_samples ({num_samples}) is greater than dataset size ({len(dataset)}), will check all samples"
            )

        return self._check_format_from_lines(dataset, num_samples)

    def _check_format_from_lines(
        self, lines: list[str], num_samples: int | None = None
    ) -> bool:
        """Check format from a list of lines.

        Args:
            lines (list[str]): List of dataset lines.
            num_samples (int | None, optional): Number of samples to check. If ``None``, checks all samples.

        Returns:
            bool: ``True`` if format is valid, ``False`` otherwise.
        """
        try:
            samples_checked = 0
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # Check if we've reached the maximum number of samples
                if num_samples is not None and samples_checked >= num_samples:
                    logger.info(
                        f"Checked {samples_checked} samples (num_samples={num_samples})"
                    )
                    break

                # Parse the line to extract problem and solution
                parsed_result = self._parse_line(line)
                if parsed_result is None:
                    logger.error(f"Line {line_num}: Failed to parse line")
                    return False

                problem, solution = parsed_result

                # Validate problem format
                if not self._validate_expression(problem):
                    logger.error(f"Line {line_num}: Invalid problem format - {problem}")
                    return False

                # Validate solution format
                if not self._validate_expression(solution):
                    logger.error(
                        f"Line {line_num}: Invalid solution format - {solution}"
                    )
                    return False

                samples_checked += 1

            return True

        except Exception as e:
            logger.error(f"Error processing dataset lines: {e}")
            return False

    def _parse_line(self, line: str) -> tuple[list[str], list[str]] | None:
        """Parse a line in the format "problem # solution".

        Args:
            line (str): Input line to parse.

        Returns:
            tuple[list[str], list[str]] | None: Tuple of (problem, solution) where each is a list of strings, or ``None`` if parsing fails.
        """
        if "#" not in line:
            logger.error(f"Line must contain '#' separator: {line}")
            return None

        try:
            problem_part, solution_part = line.split("#", 1)
            problem = self._parse_expression(problem_part.strip())
            solution = self._parse_expression(solution_part.strip())
            return problem, solution
        except Exception as e:
            logger.error(f"Error parsing line: {line}, error: {e}")
            return None

    def _parse_expression(self, expr: str) -> list[str]:
        """Parse an expression part (problem or solution) into a flat list.

        Handles formats like:
        - "x + 1" -> ["x + 1"] (single expression)
        - "x + 1 | y + 2" -> ["x + 1", "y + 2"]
        - "x + 1 | y + 2 || z + 3 | w + 4" -> ["x + 1", "y + 2", "z + 3", "w + 4"]
        - "x + 1 | y + 2 || z + 3 | w + 4 ||| a + 5 | b + 6 || c + 7 | d + 8" -> ["x + 1", "y + 2", "z + 3", "w + 4", "a + 5", "b + 6", "c + 7", "d + 8"]

        Args:
            expr (str): Expression string to parse.

        Returns:
            list[str]: Flat list of all elements.
        """
        # Check if there are any separators
        if "|" in expr:
            # Use regex to split by |, ||, |||, etc.
            # This pattern matches one or more consecutive | characters
            pattern = r"\|+"
            parts = re.split(pattern, expr)
            return [item.strip() for item in parts if item.strip()]
        else:
            # Single expression without separators
            return [expr.strip()]

    def _validate_expression(self, expr: list[str]) -> bool:
        """Validate if elements can be cast to SageMath's rings.

        Args:
            expr (list[str]): List of strings to validate.

        Returns:
            bool: ``True`` if all elements are valid, ``False`` otherwise.
        """
        try:
            return all(self._is_valid_sagemath_format(item) for item in expr)
        except (ValueError, TypeError, SyntaxError):
            return False

    def _is_valid_sagemath_format(self, s: str) -> bool:
        """Check castability to PolynomialRing or RR/QQ/ZZ.

        Args:
            s (str): String to validate.

        Returns:
            bool: ``True`` if valid SageMath expression, ``False`` otherwise.
        """
        # Remove whitespace
        s = s.strip()

        # Check for empty string
        if not s:
            return False

        try:
            # First, try polynomial rings if available
            if self.variables and self.polynomial_rings:
                for poly_ring in self.polynomial_rings:
                    try:
                        poly_ring(s)
                        return True
                    except (ValueError, TypeError, SyntaxError):
                        continue

            # Then, try as different number types
            for ring in [RR, QQ, ZZ]:
                try:
                    ring(s)
                    return True
                except (ValueError, TypeError, SyntaxError):
                    continue

            return False
        except (ValueError, TypeError, SyntaxError):
            return False
