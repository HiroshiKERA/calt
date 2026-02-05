# Polynomial reduction

多項式 f と g1,...,gt が与えられたとき、f を g1,...,gt のグレブナー基底で簡約した結果を予測するタスク。

- **係数**: 整数 ZZ
- **変数**: 3変数 (x, y, z)
- **データ**: f, g1, g2, g3 をそれぞれ 2--5 単項式で生成し、g1,g2,g3 の grevlex グレブナー基底 G を計算。f を G で簡約した余り r = I.reduce(f) のみ保存（pickle）。商は使わない（pattern 2 のときはロード時に lift で算出）。

## データ生成

```bash
cd calt/examples/polynomial_reduction
python generate_dataset.py
```

## 学習

- **order**: `grevlex`（そのまま）または `lex`（f は ring 変更、G は FGLM で lex グレブナー基底に変換）
- **pattern**: `1` = ターゲットは簡約結果のみ、`2` = ターゲットは g1..gt の係数と簡約結果

```bash
# grevlex, 簡約結果のみ
python train.py

# lex, 簡約結果のみ
python train.py --order lex

# grevlex, 係数 + 簡約結果
python train.py --pattern 2

# lex, 係数 + 簡約結果
python train.py --order lex --pattern 2
```

## ロード時の変換

- grevlex で保存した pickle を、lex で学習する場合は `PolynomialReductionLoadPreprocessor(order="lex", ...)` を使う。f は単に lex 環に移し、G は FGLM で lex 順序のグレブナー基底に変換する（零次元イデアルの場合）。
