# 開発者用
ライブラリ開発の際に読んでください

## 開発環境構築
以下のコマンドで仮想環境内にライブラリの環境を構築できる
```
uv venv # 仮想環境

source .venv/bin/activate

uv pip install --upgrade build 

python -m build

# ライブラリ本体
uv pip install dist/calt_x-0.1.0-py3-none-any.whl # versionなど適宜変更

# dev用の依存関係
uv pip install -e ".[dev]"
```

## linter + formatter
```
# linter
uv run ruff check .

# reformat
uv run ruff format .

```

## PyPIへのアップロード
下記のサイトに登録後
APIキーを発行する
TestPyPI: https://test.pypi.org/
本番PyPI: https://pypi.org/

```
# TestPyPIへのアップロード
uv pip install --upgrade twine

twine check dist/* # アップロード可能か確認

twine upload --repository testpypi dist/* # アップロード

# 本番PyPIへのアップロード
twine check dist/* 

twine upload dist/*
```

