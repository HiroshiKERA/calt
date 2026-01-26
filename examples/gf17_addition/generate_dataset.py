import random

from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline


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
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=gf17_addition_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
