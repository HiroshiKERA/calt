import numpy as np
import click
from omegaconf import OmegaConf
from sage.all import GF, ZZ
import sage.misc.randstate as randstate
from sage.misc.prandom import randint

from calt.dataset import DatasetPipeline


class ArithmeticAdditionGenerator:
    """Generate arithmetic addition (cumulative sum) samples. GF(p) or ZZ."""

    def __init__(
        self,
        field_str: str,
        min_length: int = 3,
        max_length: int = 6,
        min_value: int = 0,
        max_value: int = 100,
    ):
        self.field_str = field_str
        self.min_length = min_length
        self.max_length = max_length
        self.p = self._parse_field_str(field_str)
        self.min_value = min_value  # used only when p is None (ZZ)
        self.max_value = max_value  # used only when p is None (ZZ)

    def __call__(self, seed: int) -> tuple[str, str]:
        randstate.set_random_seed(seed)
        length = randint(self.min_length, self.max_length)

        if self.p is not None:
            K = GF(self.p)
            numbers = [K.random_element() for _ in range(length)]
            cumulative = []
            s = K(0)
            for n in numbers:
                s = s + n
                cumulative.append(s)
            numbers_int = [int(x) for x in numbers]
            cumulative_int = [int(x) for x in cumulative]
        else:
            numbers_int = [
                ZZ.random_element(self.min_value, self.max_value + 1)
                for _ in range(length)
            ]
            cumulative_int = []
            s = ZZ(0)
            for n in numbers_int:
                s = s + n
                cumulative_int.append(s)
            cumulative_int = [int(x) for x in cumulative_int]

        input_str = ",".join(map(str, numbers_int))
        output_str = ",".join(map(str, cumulative_int))
        return input_str, output_str

    def _parse_field_str(self, field_str: str) -> int | None:
        """
        Parse field_str into modulus p or None for ZZ.
        - ZZ -> p = None (use max_value for sampling range).
        - GF7, GF(7), GF37, GF(37) -> p = 7, 37, etc.
        """
        if field_str == "ZZ":
            return None
        if field_str.startswith("GF") and len(field_str) > 2:
            rest = field_str[2:].strip()
            if rest.startswith("(") and rest.endswith(")"):
                num = rest[1:-1].strip()
            else:
                num = rest
            if num.isdigit():
                return int(num)
        raise ValueError(f"Unsupported field_str: {field_str!r}")


def arithmetic_addition_stats_calc(problem: str, answer: str) -> dict[str, dict[str, int | float]]:
    return {
        "problem": _integer_list_stats(problem),
        "answer": _integer_list_stats(answer),
    }


def _integer_list_stats(data: str) -> dict[str, int | float]:
    if not data:
        raise ValueError("Cannot calculate statistics for empty data list")

    data_list = data.split(",")
    values = [int(n) for n in data_list]
    return {
        "num_values": len(data_list),
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": float(np.mean(values)),
        "std_value": float(np.std(values)),
    }


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML (problem_generator, dataset).",
)
def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    gen_cfg = OmegaConf.to_container(cfg.get("problem_generator", {}), resolve=True)
    if not gen_cfg:
        raise ValueError("config must have 'problem_generator' with field_str, min_length, max_length (and min_value, max_value for ZZ)")

    problem_generator = ArithmeticAdditionGenerator(**gen_cfg)
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
        statistics_calculator=arithmetic_addition_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
