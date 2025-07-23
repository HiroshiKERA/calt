from pathlib import Path
import yaml
import json
import pickle
import logging
from datetime import timedelta


# Type aliases for better readability
StatisticsDict = dict[str, dict[str, int | float]]
"""Type alias for statistics dictionary containing nested metrics.
Example: {"runtime": {"mean": 0.5, "std": 0.1}, "complexity": {"max": 10, "min": 1}}"""

StringProblemOrSolution = str | list[str] | list[list[str]] | list[str | list[str]]
"""Type alias for string-formatted problems and solutions.
Supports single strings, simple lists, nested lists, and mixed structures.
Maximum nesting depth is 2 levels.

Polynomial examples:
    - Single: "x^2 + 2*x + 1"
    - Simple list: ["x + y", "x - y"]
    - Nested list: [["x^2", "y^2"], ["z^2", "w^2"]]
    - Mixed: [["x", "y"], "z"] 
    
Arithmetic examples:
    - Single: "2"
    - Simple list: ["2", "3", "5", "7"]
    - Nested list: [["2", "3"], ["5", "7"]]
    - Mixed: ["2", ["3", "5"]]
"""

StringSample = tuple[StringProblemOrSolution, StringProblemOrSolution]
"""Type alias for a single problem-solution pair in string format.
Example: ("x^2 + 2*x + 1", "(x + 1)^2") or (["x + y", "x - y"], ["2*x", "2*y"])"""

StringSampleList = list[StringSample]
"""Type alias for a list of problem-solution pairs in string format.
Example: [("problem1", "solution1"), ("problem2", "solution2")]"""


class TimedeltaDumper(yaml.SafeDumper):
    """Custom YAML dumper that safely handles timedelta objects."""

    pass


def timedelta_representer(dumper: TimedeltaDumper, data: timedelta) -> yaml.ScalarNode:
    """Convert timedelta to float seconds."""
    return dumper.represent_float(data.total_seconds())


class DatasetWriter:
    """
    Dataset writer for saving problem-solution pairs in multiple formats.

    This class handles saving datasets with nested structure support up to 2 levels.
    It can save data in pickle (binary), raw text, and JSON formats.

    Attributes:
        INNER_SEP (str): Separator for single-level lists (" | ")
        OUTER_SEP (str): Separator for nested lists (" || ")
        save_dir (Path): Base directory for saving datasets
        save_text (bool): Whether to save raw text files
        save_json (bool): Whether to save JSON files
    """

    # Separator constants for string formatting
    INNER_SEP = " | "  # Separator for single-level lists
    OUTER_SEP = " || "  # Separator for nested lists

    def __init__(
        self,
        save_dir: str | None = None,
        save_text: bool = True,
        save_json: bool = True,
    ) -> None:
        """
        Initialize dataset writer.

        Args:
            save_dir: Base directory for saving datasets. If None, uses current working directory.
            save_text: Whether to save raw text files. Text files use "#" as separator
                      between problem and solution, with nested structures joined by separators.
            save_json: Whether to save JSON files. JSON files preserve the original
                      nested structure format.

        Note:
            Pickle files are always saved as they are the primary format for data loading.
            Text and JSON files are optional and controlled by save_text and save_json flags.
        """
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.save_text = save_text
        self.save_json = save_json
        self.logger = logging.getLogger(__name__)
        TimedeltaDumper.add_representer(timedelta, timedelta_representer)

    def _validate_tag(self, tag: str) -> None:
        """
        Validate tag parameter.

        Args:
            tag: Tag to validate

        Raises:
            ValueError: If tag is not 'train' or 'test'
        """
        if tag not in ("train", "test"):
            raise ValueError(f"tag must be 'train' or 'test', got '{tag}'")

    def _create_dataset_dir(self) -> Path:
        """
        Create and return dataset directory path.

        Returns:
            Path object pointing to the created dataset directory
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return self.save_dir

    def _validate_nested_structure(self, obj: StringProblemOrSolution) -> None:
        """
        Validate that the object has at most 2 levels of nesting and contains only strings.

        Args:
            obj: Object to validate

        Raises:
            ValueError: If obj contains lists deeper than 2 levels or non-string elements
        """
        if isinstance(obj, str):
            # Single string is valid
            return None
        elif isinstance(obj, list):
            # Check if this list contains any nested lists
            has_nested_lists = any(isinstance(item, list) for item in obj)

            if has_nested_lists:
                # Check that inner lists don't contain further nested lists
                for item in obj:
                    if isinstance(item, list):
                        # First check for deeper nesting (3+ levels)
                        if any(isinstance(subitem, list) for subitem in item):
                            raise ValueError(
                                f"Lists deeper than 2 levels are not supported. "
                                f"Found 3+ level structure in: {obj}"
                            )
                        # Then validate inner list contains only strings
                        if not all(isinstance(subitem, str) for subitem in item):
                            raise ValueError(
                                f"All elements in nested lists must be strings. "
                                f"Found non-string element in: {obj}"
                            )
                    else:
                        # Validate single item is string
                        if not isinstance(item, str):
                            raise ValueError(
                                f"All elements must be strings. "
                                f"Found non-string element: {item} in {obj}"
                            )
            else:
                # Simple list: validate all elements are strings
                if not all(isinstance(item, str) for item in obj):
                    raise ValueError(
                        f"All elements in simple lists must be strings. "
                        f"Found non-string element in: {obj}"
                    )
        else:
            # Neither string nor list
            raise ValueError(
                f"Object must be a string or list. "
                f"Found type {type(obj).__name__}: {obj}"
            )

    def _join_with_separators(self, obj: StringProblemOrSolution) -> str:
        """
        Join object elements with appropriate separators.

        The following rules are applied to stringify obj:
        - Single string -> return as is
        - Single-level list -> join with INNER_SEP (" | ")
        - List containing nested lists -> join each nested list with INNER_SEP (" | "),
          then join all parts with OUTER_SEP (" || ")

        Raises:
            ValueError: If obj contains lists deeper than 2 levels
        """
        # Validate structure first
        self._validate_nested_structure(obj)

        if isinstance(obj, list):
            # Check if this list contains any nested lists
            has_nested_lists = any(isinstance(item, list) for item in obj)

            if has_nested_lists:
                # Mixed structure: join with OUTER_SEP
                parts = []
                for item in obj:
                    if isinstance(item, list):
                        # Inner list: join with INNER_SEP
                        parts.append(self.INNER_SEP.join(item))
                    else:
                        # Single item: already a string
                        parts.append(item)
                return self.OUTER_SEP.join(parts)
            else:
                # Simple list: join with INNER_SEP
                return self.INNER_SEP.join(obj)
        else:
            # Single value: already a string
            return obj

    def _format_sample_strings(
        self,
        problem_str: StringProblemOrSolution,
        solution_str: StringProblemOrSolution,
    ) -> str:
        """
        Format problem and solution to string representation with separators.

        Args:
            problem_str: Problem in string format (can be nested)
            solution_str: Solution in string format (can be nested)

        Returns:
            Formatted string with problem and solution separated by " # "

        Examples:
            >>> writer._format_sample_strings("x", "y")
            "x # y"
            >>> writer._format_sample_strings(["a", "b"], ["c", "d"])
            "a | b # c | d"
            >>> writer._format_sample_strings([["a", "b"], "c"], [["d", "e"], "f"])
            "a | b || c # d | e || f"
        """
        problem_formatted = self._join_with_separators(problem_str)
        solution_formatted = self._join_with_separators(solution_str)
        return f"{problem_formatted} # {solution_formatted}"

    def _get_json_data(
        self,
        problem_str: StringProblemOrSolution,
        solution_str: StringProblemOrSolution,
    ) -> dict[str, StringProblemOrSolution]:
        """
        Format problem and solution to JSON format.

        This method creates a dictionary structure suitable for JSON serialization.
        The original nested structure is preserved exactly as provided.

        Args:
            problem_str: Problem in string format (can be nested)
            solution_str: Solution in string format (can be nested)

        Returns:
            Dictionary with "problem" and "solution" keys preserving original structure

        Examples:
            >>> writer._get_json_data("x^2 + 1", "x^2")
            {"problem": "x^2 + 1", "solution": "x^2"}
            >>> writer._get_json_data(["x + y", "x - y"], ["2*x", "2*y"])
            {"problem": ["x + y", "x - y"], "solution": ["2*x", "2*y"]}
            >>> writer._get_json_data([["x", "y"], ["z"]], [["a", "b"], ["c"]])
            {"problem": [["x", "y"], ["z"]], "solution": [["a", "b"], ["c"]]}
        """
        return {"problem": problem_str, "solution": solution_str}

    def save_batch(
        self,
        samples: StringSampleList,
        tag: str = "train",
        batch_idx: int = 0,
    ) -> None:
        """
        Save a batch of samples to files in multiple formats.

        This method saves samples in three formats:
        1. Pickle (.pkl) - Binary format, always saved, used for loading
        2. Raw text (.txt) - Human-readable format with separators (if save_text=True)
        3. JSON (.json) - Structured format preserving nested structure (if save_json=True)

        Args:
            samples: List of (problem, solution) pairs in string format
            tag: Dataset tag ("train" or "test")
            batch_idx: Batch index for incremental saving. Use 0 for first batch,
                      subsequent batches will append to existing files.

        Raises:
            ValueError: If tag is invalid or samples contain invalid nested structures

        Examples:
            >>> # Simple string samples (single problem-solution pairs)
            >>> writer = DatasetWriter(save_dir="./datasets", save_text=True, save_json=True)
            >>> samples = [
            ...     ("x^2 + 2*x + 1", "(x + 1)^2"),
            ...     ("2*x^3 - 3*x^2", "x^2*(2*x - 3)"),
            ... ]
            >>> # Creates: train_data.pkl, train_raw.txt, train_data.json
            >>> writer.save_batch(samples, tag="train", batch_idx=0)
            >>>
            >>> # 1 level nested structure samples (multiple problems/solutions)
            >>> samples = [
            ...     (["x + y", "x - y"], ["2*x", "2*y"]),
            ...     (["x^2 + y^2", "x^2 - y^2"], ["2*x^2", "2*y^2"]),
            ... ]
            >>> # Text output: "x + y | x - y # 2*x | 2*y"
            >>> writer.save_batch(samples, tag="test", batch_idx=0)
            >>>
            >>> # 2 level nested structure samples (complex nested problems)
            >>> samples = [
            ...     ([["x", "y"], ["z", "w"]], [["x", "z"], ["y", "w"]]),
            ...     ([["x + y", "x - y"], ["z + w", "z - w"]], [["x + y", "z + w"], ["x - y", "z - w"]]),
            ... ]
            >>> # Text output: "x | y || z | w # x | z || y | w"
            >>> writer.save_batch(samples, tag="test", batch_idx=0)
            >>>
            >>> # Append more samples to existing dataset
            >>> more_samples = [
            ...     ([["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]]),
            ...     ([["e", "f"], ["g", "h"]], [["e", "g"], ["f", "h"]]),
            ... ]
            >>> # Appends to existing files instead of overwriting
            >>> writer.save_batch(more_samples, tag="train", batch_idx=1)
        """
        self._validate_tag(tag)

        # Validate samples
        if not samples:
            self.logger.warning(
                "Empty samples list provided - no files will be created"
            )
            return

        dataset_dir = self._create_dataset_dir()

        # Save binary data (pickle format) - default and most efficient
        pickle_path = dataset_dir / f"{tag}_data.pkl"
        if batch_idx == 0:
            # Initialize pickle file
            with open(pickle_path, "wb") as f:
                pickle.dump(samples, f)
        else:
            # Append to pickle file
            existing_data: StringSampleList = []
            if pickle_path.exists():
                try:
                    with open(pickle_path, "rb") as f:
                        existing_data = pickle.load(f)
                except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
                    # Handle corrupted pickle files
                    self.logger.warning(f"Could not load existing pickle file: {e}")
                    pass

            existing_data.extend(samples)
            with open(pickle_path, "wb") as f:
                pickle.dump(existing_data, f)

        # Save raw text data (optional)
        if self.save_text:
            raw_path = dataset_dir / f"{tag}_raw.txt"
            mode = "w" if batch_idx == 0 else "a"
            with open(raw_path, mode) as f:
                for problem_str, solution_str in samples:
                    formatted_line = self._format_sample_strings(
                        problem_str, solution_str
                    )
                    f.write(f"{formatted_line}\n")

        # Save JSON data (optional)
        if self.save_json:
            json_path = dataset_dir / f"{tag}_data.json"
            if batch_idx == 0:
                # Initialize JSON file
                json_data = [
                    self._get_json_data(problem_str, solution_str)
                    for problem_str, solution_str in samples
                ]
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=4)
            else:
                # Append to JSON file
                existing_data: list[dict[str, StringProblemOrSolution]] = []
                if json_path.exists():
                    try:
                        with open(json_path, "r") as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        self.logger.warning(f"Could not load existing JSON file: {e}")
                        pass

                new_data = [
                    self._get_json_data(problem_str, solution_str)
                    for problem_str, solution_str in samples
                ]
                existing_data.extend(new_data)

                with open(json_path, "w") as f:
                    json.dump(existing_data, f, indent=4)

    def save_final_statistics(
        self,
        statistics: StatisticsDict,
        tag: str = "train",
    ) -> None:
        """
        Save final overall statistics to YAML file.

        Args:
            statistics: Dictionary containing dataset statistics
            tag: Dataset tag ("train" or "test")

        Raises:
            ValueError: If tag is invalid

        Note:
            Statistics are saved in YAML format for human readability.
            The file is named "{tag}_stats.yaml" in the dataset directory.
        """
        self._validate_tag(tag)
        dataset_dir = self._create_dataset_dir()

        stats_path = dataset_dir / f"{tag}_stats.yaml"
        with open(stats_path, "w") as f:
            yaml.dump(
                statistics,
                f,
                Dumper=TimedeltaDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=4,
            )

    def load_dataset(self, tag: str) -> StringSampleList:
        """
        Load dataset from pickle file.

        Args:
            tag: Dataset tag ("train" or "test")

        Returns:
            List of (problem, solution) pairs in string format

        Raises:
            ValueError: If tag is invalid
            FileNotFoundError: If the pickle file doesn't exist

        Examples:
            >>> samples = writer.load_dataset("train")
            >>> print(f"Loaded {len(samples)} samples")
            >>> for problem, solution in samples[:3]:
            ...     print(f"Problem: {problem}, Solution: {solution}")
        """
        self._validate_tag(tag)
        pickle_path = self.save_dir / f"{tag}_data.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pickle_path}")

        with open(pickle_path, "rb") as f:
            return pickle.load(f)
