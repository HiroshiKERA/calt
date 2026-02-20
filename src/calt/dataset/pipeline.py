from __future__ import annotations

from omegaconf import DictConfig

from .backends import get_backend_classes


class DatasetPipeline:
    """Pipeline for generating train/test datasets with a configurable backend.

    Uses an instance generator and optional statistics calculator to produce
    batches, then writes them to disk via the backend's DatasetWriter.
    Typically constructed via from_config() with a DictConfig (e.g. from YAML).

    Examples:
        >>> from omegaconf import OmegaConf
        >>> from calt.dataset import DatasetPipeline
        >>> cfg = OmegaConf.load("configs/dataset.yaml")
        >>> pipeline = DatasetPipeline.from_config(
        ...     cfg.dataset,
        ...     instance_generator=my_instance_generator,
        ...     statistics_calculator=my_stats_fn,
        ... )
        >>> pipeline.run()
    """

    def __init__(
        self,
        instance_generator,
        statistics_calculator,
        save_dir: str,
        save_text: bool,
        save_json: bool,
        num_train_samples: int,
        num_test_samples: int,
        batch_size: int,
        n_jobs: int,
        root_seed: int,
        verbose: bool,
        backend: str = "sagemath",
    ) -> None:
        """Initialize the dataset pipeline.

        Args:
            instance_generator: Callable that takes a single integer seed and
                returns (problem, answer). Used to generate each sample.
            statistics_calculator: Optional callable(problem, answer) returning
                a dict of per-sample statistics (e.g. {"problem": {...}, "answer": {...}}).
                Pass None to skip statistics.
            save_dir: Directory path to write dataset files.
            save_text: Whether to save samples as text files.
            save_json: Whether to save metadata (e.g. statistics) as JSON.
            num_train_samples: Number of training samples to generate.
            num_test_samples: Number of test samples to generate.
            batch_size: Number of samples per batch during generation.
            n_jobs: Number of parallel jobs for the backend generator.
            root_seed: Base seed for reproducibility; job seeds are derived from this.
            verbose: Whether to print progress.
            backend: Backend name for generation and writing ("sagemath" or "sympy").
        """
        self.instance_generator = instance_generator
        self.statistics_calculator = statistics_calculator
        self.save_dir = save_dir
        self.save_text = save_text
        self.save_json = save_json
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.root_seed = root_seed
        self.verbose = verbose
        self.backend = backend

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        instance_generator,
        statistics_calculator=None,
    ) -> "DatasetPipeline":
        """Build a DatasetPipeline from a DictConfig.

        Args:
            config: DictConfig for the dataset. Expected keys: save_dir,
                num_train_samples, num_test_samples, batch_size, n_jobs, root_seed.
                Optional (with defaults): save_text=True, save_json=True,
                verbose=True, backend="sagemath".
                Missing required keys will raise when building the pipeline.
            instance_generator: Callable(seed) -> (problem, answer). Required.
            statistics_calculator: Optional callable(problem, answer) -> dict.
                Defaults to None (no per-sample statistics).

        Returns:
            DatasetPipeline instance configured with config and the given callables.
        """
        return cls(
            instance_generator=instance_generator,
            statistics_calculator=statistics_calculator,
            save_dir=config.save_dir,
            save_text=getattr(config, "save_text", True),
            save_json=getattr(config, "save_json", True),
            num_train_samples=config.num_train_samples,
            num_test_samples=config.num_test_samples,
            batch_size=config.batch_size,
            n_jobs=config.n_jobs,
            root_seed=config.root_seed,
            verbose=getattr(config, "verbose", True),
            backend=getattr(config, "backend", "sagemath"),
        )

    def run(self) -> None:
        """Run the pipeline: generate train/test data and write to save_dir.

        Loads the backend (DatasetGenerator and DatasetWriter), then runs
        batch generation with the instance_generator and statistics_calculator,
        and writes outputs according to save_text and save_json.
        """
        DatasetGenerator, DatasetWriter = get_backend_classes(self.backend)

        # Initialize dataset generator
        dataset_generator = DatasetGenerator(
            backend="multiprocessing",
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            root_seed=self.root_seed,
        )

        # Initialize writer
        dataset_writer = DatasetWriter(
            save_dir=self.save_dir,
            save_text=self.save_text,
            save_json=self.save_json,
        )

        # Generate datasets with batch processing
        dataset_generator.run(
            dataset_sizes={
                "train": self.num_train_samples,
                "test": self.num_test_samples,
            },
            batch_size=self.batch_size,
            instance_generator=self.instance_generator,
            statistics_calculator=self.statistics_calculator,
            dataset_writer=dataset_writer,
        )
