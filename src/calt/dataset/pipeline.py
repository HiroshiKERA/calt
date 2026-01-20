from .backends import get_backend_classes

class DatasetPipeline:
    def __init__(self, problem_generator, statistics_calculator, save_dir, n_jobs, root_seed, verbose, backend="sagemath"):
        self.problem_generator = problem_generator
        self.statistics_calculator = statistics_calculator
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.root_seed = root_seed
        self.verbose = verbose

    def run(self, backend="sagemath"):
        DatasetGenerator, DatasetWriter = get_backend_classes(backend)
        
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
            save_text=True,  # whether to save raw text files
            save_json=True,  # whether to save JSON files
        )

        # Generate datasets with batch processing
        dataset_generator.run(
            dataset_sizes={"train": self.num_train_samples, "test": self.num_test_samples},
            batch_size=self.batch_size,  # set batch size
            problem_generator=self.problem_generator,
            statistics_calculator=self.statistics_calculator,
            dataset_writer=dataset_writer,
        )
