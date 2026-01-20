from calt.dataset.sagemath.dataset_generator import DatasetGenerator
from calt.dataset.sagemath.utils.dataset_writer import DatasetWriter
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def rational_factor_generator(seed: int):
    """Generate a QQ-polynomial and its factored form using PolynomialSampler."""
    import sage.misc.randstate as randstate  # type: ignore

    randstate.set_random_seed(seed)

    sampler = PolynomialSampler(
        symbols="x",
        field_str="QQ",
        order="grevlex",
        max_num_terms=3,
        max_degree=2,
        min_degree=1,
        num_bound=10,
    )

    p = sampler.sample(1)[0]
    expanded = p.expand()
    factored = p.factor()

    return str(expanded), str(factored)


if __name__ == "__main__":
    generator = DatasetGenerator(n_jobs=1, root_seed=123)
    writer = DatasetWriter(save_dir="./data", save_text=True, save_json=False)

    generator.run(
        dataset_sizes={"train": 100000, "test": 1000},
        problem_generator=rational_factor_generator,
        dataset_writer=writer,
    )
    print("Dataset generation completed")
