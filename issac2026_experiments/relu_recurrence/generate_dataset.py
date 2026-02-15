"""Generate dataset for ReLU task: rectified cumulative sum (L=10).

y_1 = x_1, and for i >= 2: y_i = ReLU(x_i + y_{i-1}).
Input: x_1,...,x_L. Target: y_1,...,y_L.
"""

import click
from omegaconf import OmegaConf
from sage.misc.prandom import randint
from sage.misc.randstate import set_random_seed

from calt.dataset import DatasetPipeline


def relu(z: float) -> float:
    return max(0.0, z)


class ReLURecurrenceGenerator:
    """ReLU recurrence: y_1 = x_1, y_i = ReLU(x_i + y_{i-1})."""

    def __init__(self, length: int = 10, x_min: int = -10, x_max: int = 10):
        """
        Args:
            length: Sequence length L.
            x_min: Min value for each x_i (inclusive).
            x_max: Max value for each x_i (inclusive).
        """
        self.length = length
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, seed: int) -> tuple[str, str]:
        set_random_seed(seed)
        x_list = [randint(self.x_min, self.x_max + 1) for _ in range(self.length)]
        y_list = [float(x_list[0])]
        for i in range(1, self.length):
            y_list.append(relu(x_list[i] + y_list[i - 1]))
        # Store as integers when possible (for cleaner text)
        y_list = [int(round(y)) if y == round(y) else y for y in y_list]
        input_str = ",".join(map(str, x_list))
        target_str = ",".join(map(str, y_list))
        return input_str, target_str


def relu_stats_calc(problem: str, answer: str) -> dict:
    return {
        "problem_len": len(problem.split(",")),
        "answer_len": len(answer.split(",")),
    }


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML.",
)
def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    gen_cfg = OmegaConf.to_container(cfg.get("problem_generator", {}), resolve=True)
    if not gen_cfg:
        raise ValueError(
            "config must have 'problem_generator' with length, x_min, x_max"
        )
    problem_generator = ReLURecurrenceGenerator(**gen_cfg)
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
        statistics_calculator=relu_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
