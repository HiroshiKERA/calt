from typing import Any
from pathlib import Path
import yaml
import json
import pickle
from datetime import timedelta


# Type aliases for better readability
Sample = tuple[list[Any] | Any, list[Any] | Any]
SampleList = list[Sample]
StatisticsDict = dict[str, Any]
JsonSample = dict[str, list[str] | str]
JsonData = list[JsonSample]


class TimedeltaDumper(yaml.SafeDumper):
    """Custom YAML dumper that safely handles timedelta objects."""

    pass


def timedelta_representer(dumper: TimedeltaDumper, data: timedelta) -> yaml.ScalarNode:
    """Convert timedelta to float seconds."""
    return dumper.represent_float(data.total_seconds())


class DatasetWriter:
    def __init__(
        self,
        save_dir: str | None = None,
        save_text: bool = True,
        save_json: bool = True,
    ) -> None:
        """
        Initialize dataset writer.

        Args:
            save_dir: Base directory for saving datasets
            save_text: Whether to save raw text files (optional)
            save_json: Whether to save JSON files (optional)
        """
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.save_text = save_text
        self.save_json = save_json
        TimedeltaDumper.add_representer(timedelta, timedelta_representer)

    def _validate_tag(self, tag: str) -> None:
        """Validate tag parameter."""
        if tag not in ("train", "test"):
            raise ValueError(f"tag must be 'train' or 'test', got '{tag}'")

    def _get_dataset_dir(self, data_tag: str | None = None) -> Path:
        """Get dataset directory path."""
        if data_tag:
            return self.save_dir / f"dataset_{data_tag}"
        return self.save_dir

    def _ensure_dir(self, data_tag: str | None = None) -> Path:
        """Ensure dataset directory exists."""
        dataset_dir = self._get_dataset_dir(data_tag)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _convert_sample_to_strings(
        self, problem: list[Any] | Any, solution: list[Any] | Any
    ) -> tuple[str, str]:
        """Convert problem and solution to string representations."""
        problem_str = (
            " | ".join(str(p) for p in problem)
            if isinstance(problem, list)
            else str(problem)
        )
        solution_str = (
            " | ".join(str(s) for s in solution)
            if isinstance(solution, list)
            else str(solution)
        )
        return problem_str, solution_str

    def _convert_sample_to_json(
        self, problem: list[Any] | Any, solution: list[Any] | Any
    ) -> JsonSample:
        """Convert problem and solution to JSON format."""
        problem_str = (
            [str(p) for p in problem] if isinstance(problem, list) else str(problem)
        )
        solution_str = (
            [str(s) for s in solution] if isinstance(solution, list) else str(solution)
        )
        return {"problem": problem_str, "solution": solution_str}

    def save_batch(
        self,
        samples: SampleList,
        tag: str = "train",
        batch_idx: int = 0,
        data_tag: str | None = None,
    ) -> None:
        """Save a batch of samples to files."""
        self._validate_tag(tag)
        dataset_dir = self._ensure_dir(data_tag)

        # Save binary data (pickle format) - default and most efficient
        pickle_path = dataset_dir / f"{tag}_data.pkl"
        if batch_idx == 0:
            # Initialize pickle file
            with open(pickle_path, "wb") as f:
                pickle.dump(samples, f)
        else:
            # Append to pickle file
            existing_data: SampleList = []
            if pickle_path.exists():
                try:
                    with open(pickle_path, "rb") as f:
                        existing_data = pickle.load(f)
                except (pickle.UnpicklingError, FileNotFoundError):
                    pass

            existing_data.extend(samples)
            with open(pickle_path, "wb") as f:
                pickle.dump(existing_data, f)

        # Save raw text data (optional)
        if self.save_text:
            raw_path = dataset_dir / f"{tag}_raw.txt"
            mode = "w" if batch_idx == 0 else "a"
            with open(raw_path, mode) as f:
                for problem, solution in samples:
                    problem_str, solution_str = self._convert_sample_to_strings(
                        problem, solution
                    )
                    f.write(f"{problem_str} # {solution_str}\n")

        # Save JSON data (optional)
        if self.save_json:
            json_path = dataset_dir / f"{tag}_data.json"
            if batch_idx == 0:
                # Initialize JSON file
                json_data: JsonData = [
                    self._convert_sample_to_json(problem, solution)
                    for problem, solution in samples
                ]
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=4)
            else:
                # Append to JSON file
                existing_data: JsonData = []
                if json_path.exists():
                    try:
                        with open(json_path, "r") as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass

                new_data: JsonData = [
                    self._convert_sample_to_json(problem, solution)
                    for problem, solution in samples
                ]
                existing_data.extend(new_data)

                with open(json_path, "w") as f:
                    json.dump(existing_data, f, indent=2)

    def save_final_statistics(
        self,
        statistics: StatisticsDict,
        tag: str = "train",
        data_tag: str | None = None,
    ) -> None:
        """Save final overall statistics."""
        self._validate_tag(tag)
        dataset_dir = self._ensure_dir(data_tag)

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

    def load_dataset(self, tag: str, data_tag: str | None = None) -> SampleList:
        """
        Load dataset from binary file.

        Args:
            tag: Dataset tag ("train" or "test")
            data_tag: Optional tag for the dataset directory

        Returns:
            List of (problem, solution) pairs
        """
        self._validate_tag(tag)
        dataset_dir = self._get_dataset_dir(data_tag)
        pickle_path = dataset_dir / f"{tag}_data.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pickle_path}")

        with open(pickle_path, "rb") as f:
            return pickle.load(f)
