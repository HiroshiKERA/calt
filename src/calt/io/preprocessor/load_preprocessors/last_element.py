"""Last-element load preprocessor for cumulative-sum style tasks."""

from typing import Any

from ..load_preprocessor import _get_answer_from_source, _to_str


class LastElementLoadPreprocessor:
    """Use only the last element of answer (e.g. cumulative-sum final value).

    - テキスト行: ``\"11,4,11,4 # 11,15,9,13\"`` のような 1 行 (format: problem # answer)
    - JSONL: ``{\"problem\": ..., \"answer\": ...}`` (or \"solution\") 形式の dict
    - ``answer`` は以下のいずれか:
      - リスト (例: ``[11, 15, 9, 13]``)
      - 区切り文字で連結された文字列 (例: ``\"11,15,9,13\"``)
    - 出力は ``(input_text, last_answer_str)`` で、最後の要素だけをターゲットにする。
      例: ``\"11,4,11,4 # 11,15,9,13\"`` → 入力: ``\"11,4,11,4\"``, ターゲット: ``\"13\"``
    """

    def __init__(self, problem_to_str: Any = None, delimiter: str = ","):
        # problem 側のフォーマットは既存の _to_str に任せる
        self.problem_to_str = problem_to_str or _to_str
        self.delimiter = delimiter

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        # テキスト行 ("11,4,11,4 # 11,15,9,13") の場合 (format: problem # answer)
        if isinstance(source, str):
            line = source.strip()
            if "#" not in line:
                raise ValueError(
                    f"LastElementLoadPreprocessor: expected '#' delimiter in text line, got: {line!r}"
                )
            problem_str, answer_str = line.split("#", 1)
            input_text = problem_str.strip()
            s = answer_str.strip()
            if self.delimiter in s:
                tokens = [tok.strip() for tok in s.split(self.delimiter) if tok.strip()]
                last = tokens[-1] if tokens else s
            else:
                last = s
            target_text = last
            return input_text, target_text

        # JSONL / pickle の dict 形式 {"problem": ..., "answer": ...} (or "solution") の場合
        if not isinstance(source, dict):
            raise TypeError("LastElementLoadPreprocessor expects str or dict source")

        problem = source.get("problem")
        answer = _get_answer_from_source(source)
        if problem is None or answer is None:
            raise ValueError(
                "Source must have 'problem' and 'answer' (or 'solution') keys"
            )

        # 入力テキストはそのまま（または _to_str で整形）
        input_text = self.problem_to_str(problem)

        # answer がリストの場合: その最後の要素
        if isinstance(answer, list) and answer:
            last = answer[-1]
        # answer が区切り文字で連結された文字列の場合: split して最後のトークン
        elif isinstance(answer, str):
            s = answer.strip()
            if self.delimiter in s:
                tokens = [tok.strip() for tok in s.split(self.delimiter) if tok.strip()]
                last = tokens[-1] if tokens else s
            else:
                last = s
        else:
            # それ以外はそのまま1個の値とみなす
            last = answer

        target_text = _to_str(last) if not isinstance(last, str) else last
        return input_text, target_text
