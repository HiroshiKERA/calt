"""Generate dataset for Groebner task (2-variable, F=[f1,f2] -> G=Groebner basis).

参考: /home/kera/workspace/Transformer-GB/src/dataset/groebner.sage,
      /home/kera/workspace/Transformer-GB/src/dataset/build_dataset.sage.py

ここでは簡易版として:
  - 2変数多項式リング R (symbols, field_str, order は config から取得)
  - F = [f1, f2] を PolynomialSampler でサンプル
  - I = (F) の Groebner 基底 G を SageMath で計算 (libsingular:stdfglm)
  - テキスト形式: input = \"f1 | f2\", target = \"g1 | g2 | ...\"
将来的に monomial order 変換や C/E 形式などは load 時の preprocessor で調整する想定。
"""

import click
from omegaconf import OmegaConf
import sage.misc.randstate as randstate  # type: ignore

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


class GroebnerGenerator:
    """Problem generator: F=[f1,...], Solution: Groebner basis G of ideal (F)."""

    def __init__(self, sampler: PolynomialSampler, num_polynomials: int = 2):
        self.sampler = sampler
        self.num_polynomials = num_polynomials

    def __call__(self, seed: int) -> tuple[str, str]:
        randstate.set_random_seed(seed)

        R = self.sampler.get_ring()
        # 多項式のリスト F をサンプル（デフォルトは 2 本）
        F = self.sampler.sample(num_samples=self.num_polynomials)
        I = R.ideal(F)

        # Groebner 基底の計算
        # libsingular:stdfglm は環やバージョン依存でエラーを吐きやすいので、
        # まずは Sage のデフォルトアルゴリズムで計算する。
        G = list(I.groebner_basis())

        # problem_str = " | ".join(str(f) for f in F)
        # solution_str = " | ".join(str(g) for g in G)
        return F, G

@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML (problem_generator, dataset).",
)
def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    # sampler 設定から PolynomialSampler を構築（他の多項式タスクと同じパターン）
    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    sampler = PolynomialSampler(**sampler_cfg)

    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True)
    problem_generator = GroebnerGenerator(sampler=sampler, **gen_cfg)
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()

