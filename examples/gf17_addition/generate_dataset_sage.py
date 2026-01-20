import random
from calt.dataset.sagemath.dataset_generator import DatasetGenerator
from calt.dataset.sagemath.utils.dataset_writer import DatasetWriter


def gf17_addition_generator(seed):
    random.seed(seed)
    p = 17

    length = random.randint(3, 6)
    numbers = [random.randint(0, p - 1) for _ in range(length)]

    cumulative = []
    s = 0
    for n in numbers:
        s = (s + n) % p
        cumulative.append(s)

    input_str = ",".join(map(str, numbers))
    output_str = ",".join(map(str, cumulative))

    return f"{input_str}", f"{output_str}"


if __name__ == "__main__":
    generator = DatasetGenerator(n_jobs=1, root_seed=100)
    writer = DatasetWriter(save_dir="./data", save_text=True, save_json=False)

    generator.run(
        dataset_sizes={"train": 100000, "test": 1000},
        problem_generator=gf17_addition_generator,
        dataset_writer=writer,
    )
    print("Dataset generation completed")
