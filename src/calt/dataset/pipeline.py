from __future__ import annotations

from omegaconf import DictConfig

from .backends import get_backend_classes


class DatasetPipeline:
    """Pipeline for generating datasets.

    Example:
        >>> from omegaconf import OmegaConf
        >>> from calt.dataset.pipeline import DatasetPipeline
        >>> cfg = OmegaConf.load("configs/dataset.yaml")
        >>> pipeline = DatasetPipeline.from_config(
        ...     cfg.dataset,
        ...     problem_generator=my_problem_generator,
        ...     statistics_calculator=my_stats_fn,
        ... )
        >>> pipeline.run()
    """

    def __init__(
        self,
        problem_generator,
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
        self.problem_generator = problem_generator
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
        problem_generator,
        statistics_calculator,
    ) -> "DatasetPipeline":
        """Create a DatasetPipeline from a DictConfig."""
        return cls(
            problem_generator=problem_generator,
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
            dataset_sizes={"train": self.num_train_samples, "test": self.num_test_samples},
            batch_size=self.batch_size,  # set batch size
            problem_generator=self.problem_generator,
            statistics_calculator=self.statistics_calculator,
            dataset_writer=dataset_writer,
        )
