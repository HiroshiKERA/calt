from typing import Any, List, Tuple, Optional, Dict, Union
from pathlib import Path
import yaml
import json
from datetime import timedelta


class TimedeltaDumper(yaml.SafeDumper):
    """Custom YAML dumper that safely handles timedelta objects."""

    pass


def timedelta_representer(dumper: TimedeltaDumper, data: timedelta) -> yaml.ScalarNode:
    """Convert timedelta to float seconds."""
    return dumper.represent_float(data.total_seconds())


class DatasetWriter:
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize dataset writer.

        Args:
            save_dir: Base directory for saving datasets. If None, uses current directory.
        """
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        # Register the timedelta representer
        TimedeltaDumper.add_representer(timedelta, timedelta_representer)

    def save_dataset(
        self,
        samples: List[Tuple[Union[List[Any], Any], Union[List[Any], Any]]],
        statistics: Dict[str, Any],
        tag: str = "train",
        data_tag: Optional[str] = None,
    ) -> None:
        """
        Save the generated dataset and its statistics.

        Args:
            samples: List of (F, G) pairs where F and G are either lists of polynomials or single polynomials
            statistics: Dictionary containing dataset statistics
            tag: Dataset tag (e.g., "train", "test", "valid")
            data_tag: Optional tag for the dataset directory
        """
        # Create dataset directory
        dataset_dir = self._get_dataset_dir(data_tag)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics in YAML format
        stats_path = dataset_dir / f"{tag}_stats.yaml"
        with open(stats_path, "w") as f:
            yaml.dump(
                statistics,
                f,
                Dumper=TimedeltaDumper,
                default_flow_style=False,
                sort_keys=False,
            )

        # Save raw data in text format
        raw_path = dataset_dir / f"{tag}_raw.txt"
        self._save_raw(samples, raw_path)

        # Save data in JSON format
        json_path = dataset_dir / f"{tag}_data.json"
        self._save_json(samples, json_path)

    def _get_dataset_dir(self, tag: Optional[str] = None) -> Path:
        """
        Get directory path for dataset based on field and number of variables.

        Args:
            tag: Optional tag to add to directory name

        Returns:
            Path object for the dataset directory
        """
        if tag:
            return self.save_dir / f"dataset_{tag}"
        return self.save_dir

    def _save_raw(
        self,
        samples: List[Tuple[Union[List[Any], Any], Union[List[Any], Any]]],
        base_path: Path,
    ) -> None:
        """
        Save polynomial systems in text format, which is handy for SageMath.
        Format: f1 | f2 | ... | fs # g1 | g2 | ... | gt

        Args:
            samples: List of (F, G) pairs where F and G are either lists of polynomials or single polynomials
            base_path: Base path for the output file
        """
        with open(base_path, "w") as f:
            for F, G in samples:
                # Convert polynomials to strings and join with |
                if isinstance(F, list):
                    f_str = " | ".join(str(p) for p in F)
                else:
                    f_str = str(F)
                if isinstance(G, list):
                    g_str = " | ".join(str(p) for p in G)
                else:
                    g_str = str(G)
                    
                f.write(f"{f_str} # {g_str}\n")

    def _save_json(
        self,
        samples: List[Tuple[Union[List[Any], Any], Union[List[Any], Any]]],
        base_path: Path,
    ) -> None:
        """
        Save polynomial systems in JSON format.

        Args:
            samples: List of (F, G) pairs where F and G are either lists of polynomials or single polynomials
            base_path: Base path for the output file
        """
        json_data = []
        for F, G in samples:
            if isinstance(F, list):
                input_data = [str(p) for p in F]
            else:
                input_data = str(F)
            if isinstance(G, list):
                output_data = [str(p) for p in G]
            else:
                output_data = str(G)
            json_data.append({"input": input_data, "output": output_data})

        with open(base_path, "w") as f:
            json.dump(json_data, f, indent=2)
