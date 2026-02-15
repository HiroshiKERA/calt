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
import sage.misc.randstate as randstate  # type: ignore
from omegaconf import OmegaConf
from sage.rings.rational_field import QQ  # type: ignore

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def _has_large_rational_coefficients(polynomials, threshold: int = 100) -> bool:
    """QQ の場合に，係数が「分子または分母の絶対値が threshold 以上」の有理数を含むか判定する."""

    for p in polynomials:
        # p.coefficients() は base_ring の要素（QQ の場合は有理数）を返す
        for c in p.coefficients():
            try:
                num = c.numerator()
                den = c.denominator()
            except Exception:
                # QQ 以外（GF, ZZ など）の係数型では何もしない
                continue
            if abs(int(num)) >= threshold or abs(int(den)) >= threshold:
                return True
    return False


class GroebnerGenerator:
    """Problem generator: F=[f1,...], Solution: Groebner basis G of ideal (F)."""

    def __init__(self, sampler: PolynomialSampler, num_polynomials: int = 2):
        self.sampler = sampler
        self.num_polynomials = num_polynomials

    def __call__(self, seed: int) -> tuple[str, str]:
        randstate.set_random_seed(seed)

        R = self.sampler.get_ring()
        is_QQ = R.base_ring() == QQ

        # QQ の場合のみ，「分子・分母が3桁以上の有理数係数を含むインスタンス」を捨てて再サンプルする
        max_retries = 100
        last_F = None
        last_G = None
        for _ in range(max_retries):
            # 多項式のリスト F をサンプル（デフォルトは 2 本）
            F = self.sampler.sample(num_samples=self.num_polynomials)
            ideal = R.ideal(F)

            # Groebner 基底の計算
            # libsingular:stdfglm は環やバージョン依存でエラーを吐きやすいので、
            # まずは Sage のデフォルトアルゴリズムで計算する。
            G = list(ideal.groebner_basis())

            last_F, last_G = F, G

            if not is_QQ:
                # QQ 以外（GF7, ZZ など）はそのまま採用
                break

            # QQ のとき，F と G の係数をチェック
            if not _has_large_rational_coefficients(list(F) + list(G), threshold=100):
                break

        # 失敗時も最後に得られた F, G を返す
        return last_F, last_G


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
