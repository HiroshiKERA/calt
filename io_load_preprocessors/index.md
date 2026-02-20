# Load preprocessors

Load preprocessors run once at load time (before the lexer) to convert raw file content—a text line, JSONL object, or pickle sample—into `(input_text, target_text)` pairs. You can use the library-provided implementations or supply your own and pass them to [IOPipeline](../io_pipeline/) via configuration.

## ChainLoadPreprocessor

```
ChainLoadPreprocessor(*preprocessors: Any)
```

Run multiple load preprocessors in sequence.

The first receives the raw source (str or dict); each next receives the previous output. The last preprocessor must return (input_text, target_text). Use this to combine e.g. TextToSageLoadPreprocessor and ExpandedFormLoadPreprocessor.

Source code in `src/calt/io/preprocessor/load_preprocessors/chain.py`

```
def __init__(self, *preprocessors: Any):
    if not preprocessors:
        raise ValueError("ChainLoadPreprocessor requires at least one preprocessor")
    self.preprocessors = list(preprocessors)
```

## ExpandedFormLoadPreprocessor

```
ExpandedFormLoadPreprocessor(delimiter: str = ' || ')
```

Convert pickle-loaded polynomials to C/E expanded form (input_text, target_text).

Expects source to be a dict with "problem" and "answer" (or "solution") (as from pickle or JSONL that stored raw polynomial objects). Problem and answer can be a single polynomial or a list of polynomials. Each polynomial is converted to: "C E E ... + C E ..." Multiple polynomials are joined by delimiter (default " || ").

Source code in `src/calt/io/preprocessor/load_preprocessors/expanded_form.py`

```
def __init__(self, delimiter: str = " || "):
    self.delimiter = delimiter
```

## TextToSageLoadPreprocessor

```
TextToSageLoadPreprocessor(
    delimiter: str = " | ", ring: Callable[[str], Any] | None = None
)
```

Parse text line into SageMath polynomial lists (dict for chaining).

Expects source to be a string line: "poly1 | poly2 # poly3 | poly4" (problem # answer). Splits by delimiter to get polynomial strings, parses each with ring(poly_str), returns {"problem": [poly, ...], "answer": [poly, ...]} for use with ExpandedFormLoadPreprocessor (e.g. via ChainLoadPreprocessor).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delimiter` | `str` | Separator between polynomials in input text (default " | "). | `' | '` | | `ring` | `Callable[[str], Any] | None` | Callable that takes a string and returns a polynomial (e.g. SageMath polynomial ring R so that R("9x0 + 5x2 + 10") works). | `None` |

Source code in `src/calt/io/preprocessor/load_preprocessors/text_to_sage.py`

```
def __init__(
    self,
    delimiter: str = " | ",
    ring: Callable[[str], Any] | None = None,
):
    if ring is None:
        raise ValueError("TextToSageLoadPreprocessor requires ring")
    self.delimiter = delimiter
    self.ring = ring
```

## ReversedOrderLoadPreprocessor

```
ReversedOrderLoadPreprocessor(problem_to_str: Any = None, delimiter: str = ',')
```

Reverse the order of answer elements (split by delimiter, reverse, rejoin).

- Text line: `"11,4,11,4 # 11,15,9,13"` → input: `"11,4,11,4"`, target: `"13,9,15,11"`
- JSONL: same for `{"problem": ..., "answer": ...}` (or "solution"); split answer by delimiter, reverse, rejoin.

Source code in `src/calt/io/preprocessor/load_preprocessors/reversed_order.py`

```
def __init__(self, problem_to_str: Any = None, delimiter: str = ","):
    self.problem_to_str = problem_to_str or _to_str
    self.delimiter = delimiter
```

## LastElementLoadPreprocessor

```
LastElementLoadPreprocessor(problem_to_str: Any = None, delimiter: str = ',')
```

Use only the last element of answer (e.g. cumulative-sum final value).

- Text line: single line like `"11,4,11,4 # 11,15,9,13"` (format: problem # answer)
- JSONL: dict with `{"problem": ..., "answer": ...}` (or "solution")
- `answer` is one of:
- list (e.g. `[11, 15, 9, 13]`)
- delimiter-joined string (e.g. `"11,15,9,13"`)
- Output is `(input_text, last_answer_str)`; only the last element is used as target. e.g. `"11,4,11,4 # 11,15,9,13"` → input: `"11,4,11,4"`, target: `"13"`

Source code in `src/calt/io/preprocessor/load_preprocessors/last_element.py`

```
def __init__(self, problem_to_str: Any = None, delimiter: str = ","):
    # Problem formatting is delegated to existing _to_str
    self.problem_to_str = problem_to_str or _to_str
    self.delimiter = delimiter
```
