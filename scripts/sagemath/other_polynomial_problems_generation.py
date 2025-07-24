from calt import (
    PolynomialSampler,
    DatasetGenerator,
    DatasetWriter,
)
from other_polynomial_problems import (
    SumProblemGenerator,
    GCDProblemGenerator,
    ProductProblemGenerator,
    PartialProdProblemGenerator,
)
from polynomial_problem_generation import (
    PolyStatisticsCalculator,
    PartialSumProblemGenerator,
)


def get_symbols(nvars: int) -> str:
    return ", ".join([f"x{i}" for i in range(nvars)])


def generate_dataset(
    problem_type: str,
    save_dir: str,
    field_str: str,
    nvars: int,
    num_train_samples: int = 100000,
    num_test_samples: int = 1000,
    max_polynomials: int = 5,
    min_polynomials: int = 2,
    max_num_terms: int = 5,
    max_degree: int = 10,
    min_degree: int = 1,
):
    """
    Generate dataset for a specific problem type.

    Args:
        problem_type: Type of problem ('sum', 'gcd', 'product', 'partial_prod')
        save_dir: Directory to save the dataset
        num_train_samples: Number of training samples
        num_test_samples: Number of test samples
        max_polynomials: Maximum number of polynomials in input
        min_polynomials: Minimum number of polynomials in input
        max_num_terms: Maximum number of terms in each polynomial
        max_degree: Maximum degree of each polynomial
        min_degree: Minimum degree of each polynomial
    """
    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        symbols=get_symbols(nvars),
        field_str=field_str,
        order="grevlex",
        max_num_terms=max_num_terms,
        max_degree=max_degree,
        min_degree=min_degree,
        degree_sampling="uniform",
        term_sampling="uniform",
        max_coeff=None,
        num_bound=None,
        strictly_conditioned=True,
        nonzero_instance=True,
        max_attempts=1000,
    )

    # Initialize problem generator based on problem type
    if problem_type == "sum":
        problem_generator = SumProblemGenerator(
            sampler=sampler,
            max_polynomials=max_polynomials,
            min_polynomials=min_polynomials,
        )
    elif problem_type == "partial_sum":
        problem_generator = PartialSumProblemGenerator(
            sampler=sampler,
            max_polynomials=max_polynomials,
            min_polynomials=min_polynomials,
        )
    elif problem_type == "gcd":
        problem_generator = GCDProblemGenerator(sampler=sampler)
    elif problem_type == "product":
        problem_generator = ProductProblemGenerator(
            sampler=sampler,
            max_polynomials=max_polynomials,
            min_polynomials=min_polynomials,
        )
    elif problem_type == "partial_prod":
        problem_generator = PartialProdProblemGenerator(
            sampler=sampler,
            max_polynomials=max_polynomials,
            min_polynomials=min_polynomials,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Initialize statistics calculator
    statistics_calculator = PolyStatisticsCalculator()

    # Initialize dataset generator
    dataset_generator = DatasetGenerator(
        backend="multiprocessing",
        n_jobs=-1,
        verbose=True,
        root_seed=100,
    )

    # Initialize writer
    dataset_writer = DatasetWriter(
        save_dir=save_dir,
        save_text=True,  # whether to save raw text files
        save_json=True,  # whether to save JSON files
    )

    # Generate training set
    dataset_generator.run(
        dataset_sizes={"train": num_train_samples, "test": num_test_samples},
        batch_size=100000,  # set batch size
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
        dataset_writer=dataset_writer,
    )


def main():
    # Generate datasets for each problem type
    problem_types = ["sum", "partial_sum", "gcd", "product", "partial_prod"]
    field_strs = ["GF7", "RR", "ZZ", "QQ"]
    nvars = 2

    for problem_type in problem_types:
        for field_str in field_strs:
            save_dir = f"dataset/sympy/{problem_type}_problem/{field_str}_n={nvars}"
            print(
                f"=============== Generating dataset for {problem_type} problem ({field_str}, nvars={nvars}) ===============\n"
            )
            generate_dataset(
                problem_type,
                save_dir,
                field_str=field_str,
                nvars=nvars,
                num_train_samples=1000,
                num_test_samples=1000,
            )
            print(
                f"\n=============== Dataset for {problem_type} problem generated successfully! ===============\n"
            )
            print()


if __name__ == "__main__":
    main()
