import os
import logging
from sage.all import PolynomialRing, QQ, RR, ZZ

# Set up logger for this module
logger = logging.getLogger(__name__)


class FormatChecker:
    """
    Format checker for SageMath dataset validation.

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
        variable_name: str = "x",
    ):
        """
        Initialize the format checker.

        Args:
            variable_names: Comma-separated string of variable names (e.g., "x, y, z").
            num_vars: Number of variables to generate (this is required, if variable_names is not specified).
            variable_name: Base name for variables when using num_vars (this is required, if num_vars is specified).

        Raises:
            ValueError: If both variable_names and num_vars are specified, or if num_vars is specified without variable_name.
        """
        # Validate parameter combinations
        if variable_names is not None and num_vars is not None:
            raise ValueError(
                "Cannot specify both 'variable_names' and 'num_vars'. Use either 'variable_names' or ('num_vars' and 'variable_name')."
            )

        if num_vars is not None and variable_name is None:
            raise ValueError("'num_vars' requires 'variable_name' to be specified.")

        # Set variables based on parameters
        if variable_names is not None:
            # Parse comma-separated variable names into a list
            self.variables = [
                var.strip() for var in variable_names.split(",") if var.strip()
            ]
            if not self.variables:
                raise ValueError("No valid variable names found in 'variable_names'")
        elif num_vars is not None:
            self.variables = [f"{variable_name}{i}" for i in range(num_vars)]
        else:
            self.variables = None

    def check_format(self, dataset_path: str, max_samples: int | None = None) -> bool:
        """
        Check if the dataset follows the correct SageMath format.

        This method validates that all problems and solutions in the dataset can be
        properly parsed and cast to SageMath's mathematical structures (PolynomialRing, RR, QQ, ZZ).
        It is designed to be run before using the dataset for model training or evaluation
        to ensure data quality and prevent runtime errors.

        Args:
            dataset_path: Path to the dataset file
            max_samples: Maximum number of samples to check. If None, checks all samples.

        Returns:
            True if format is valid, False otherwise

        Example:
            >>> # variable_names is specified
            >>> checker = FormatChecker(variable_names="x, y, z")
            >>> checker.check_format("dataset/test.txt")
            True
            >>> # num_vars and variable_name are specified
            >>> checker = FormatChecker(num_vars=3, variable_name="x")
            >>> checker.check_format("dataset/test.txt")
            True
            >>> # Check only first 10 samples
            >>> checker = FormatChecker(num_vars=3, variable_name="x")
            >>> checker.check_format("dataset/test.txt", max_samples=10)
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
        # Check if file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_path, "r") as f:
                samples_checked = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Check if we've reached the maximum number of samples
                    if max_samples is not None and samples_checked >= max_samples:
                        logger.info(
                            f"Checked {samples_checked} samples (max_samples={max_samples})"
                        )
                        break

                    # Parse the line to extract problem and solution
                    problem, solution = self._parse_line(line)

                    # Validate problem format
                    if not self._validate_expression(problem):
                        logger.error(
                            f"Line {line_num}: Invalid problem format - {problem}"
                        )
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
            logger.error(f"Error reading dataset: {e}")
            return False

    def _parse_line(self, line: str) -> tuple[list[str], list[str]]:
        """
        Parse a line in the format "problem # solution".

        Args:
            line: Input line to parse

        Returns:
            Tuple of (problem, solution) where each is a list of strings
        """
        if "#" not in line:
            raise ValueError("Line must contain '#' separator")

        problem_part, solution_part = line.split("#", 1)
        problem = self._parse_expression(problem_part.strip())
        solution = self._parse_expression(solution_part.strip())

        return problem, solution

    def _parse_expression(self, expr: str) -> list[str]:
        """
        Parse an expression part (problem or solution) into a flat list.

        Handles formats like:
        - "x + 1" -> ["x + 1"] (single expression)
        - "x + 1 | y + 2" -> ["x + 1", "y + 2"]
        - "x + 1 | y + 2 || z + 3 | w + 4" -> ["x + 1", "y + 2", "z + 3", "w + 4"]
        - "x + 1 | y + 2 || z + 3 | w + 4 ||| a + 5 | b + 6 || c + 7 | d + 8" -> ["x + 1", "y + 2", "z + 3", "w + 4", "a + 5", "b + 6", "c + 7", "d + 8"]

        Args:
            expr: Expression string to parse

        Returns:
            Flat list of all elements
        """
        # Check if there are any separators
        if "|" in expr:
            # Replace ||| and || with |, then split by |
            flat_expr = expr.replace("|||", "|").replace("||", "|")
            return [item.strip() for item in flat_expr.split("|") if item.strip()]
        else:
            # Single expression without separators
            return [expr.strip()]

    def _validate_expression(self, expr: list[str]) -> bool:
        """
        Validate if all elements in the expression can be cast to SageMath's PolynomialRing or any of RR, QQ, ZZ.

        Args:
            expr: List of strings to validate

        Returns:
            True if all elements are valid, False otherwise
        """
        try:
            return all(self._is_valid_sagemath_string(item) for item in expr)
        except (ValueError, TypeError, SyntaxError):
            return False

    def _is_valid_sagemath_string(self, s: str) -> bool:
        """
        Check if a string can be cast to SageMath's PolynomialRing or any of RR, QQ, ZZ.

        Args:
            s: String to validate

        Returns:
            True if valid SageMath expression, False otherwise
        """
        # Remove whitespace
        s = s.strip()

        # Check for empty string
        if not s:
            return False

        try:
            if self.variables:
                # Try different coefficient rings for PolynomialRing
                for ring in [RR, QQ, ZZ]:
                    try:
                        R = PolynomialRing(ring, self.variables)
                        R(s)
                        return True
                    except (ValueError, TypeError, SyntaxError):
                        continue
                return False
            else:
                # Try as different number types
                for ring in [RR, QQ, ZZ]:
                    try:
                        ring(s)
                        return True
                    except (ValueError, TypeError, SyntaxError):
                        continue
                return False
        except (ValueError, TypeError, SyntaxError):
            return False
