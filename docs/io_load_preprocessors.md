# Load preprocessors

Load preprocessors run once at load time (before the lexer) to convert raw file content—a text line, JSONL object, or pickle sample—into `(input_text, target_text)` pairs. You can use the library-provided implementations or supply your own and pass them to [IOPipeline](io_pipeline.md) via configuration.

A function wrapped in `UserCallableLoadPreprocessor` may also return a third element, a dict of extra labels (`return input_text, target_text, {"carry_labels": [...]}`). They are batched and passed to a custom loss by name; see [Custom losses and auxiliary tasks](model_pipeline.md#custom-losses-and-auxiliary-tasks) for their shapes, the length rule, and the fill value.

## Lists in text and JSONL data

`DatasetWriter` can save each sample both as a text line and as a JSONL object, but the two are not read into the same text by default, so do not treat them as interchangeable data:

- **JSONL** (and pickle) keeps the problem and the answer as they were before formatting, e.g. the list `[["1", "2"], ["3", "4"]]`.
- **Text** holds the formatted result: `DatasetWriter` joins a flat list with `" | "`, and a list of lists with `" | "` inside and `" || "` outside, so the same sample is the line `1 | 2 || 3 | 4`. Reading text uses that string as it is.
- Reading JSONL with the default loader joins every level with a space, giving `1 2 3 4`. Whitespace is not a token, so after tokenization the boundaries between the elements are gone (with `digit_group: 1`, `["31600", "54662"]` becomes `3 1 6 0 0 5 4 6 6 2`).

### `list_separators`

`data.list_separators` (or `IOPipeline(..., list_separators=...)`, or the `list_separators` argument of `JsonlDefaultLoadPreprocessor` and `PickleDefaultLoadPreprocessor`) makes the default JSONL and pickle loaders put separators between list elements:

```yaml
data:
  train_dataset_path: ../data/train_data.jsonl
  test_dataset_path: ../data/test_data.jsonl
  lexer_config: ../configs/lexer.yaml
  list_separators: ["||", "|"]   # quote them: a bare | is YAML syntax
```

- Index 0 is used for the outermost list, index 1 for the lists inside it, and so on. Each separator is put between spaces. With `["||", "|"]`, `[["1", "2"], ["3", "4"]]` becomes `1 | 2 || 3 | 4`; with `["|"]`, `["31600", "54662"]` becomes `31600 | 54662`.
- One separator on a nested list applies to the outermost list only, and the inner lists are joined with spaces (`["|"]`: `1 2 | 3 4`). Several separators on a flat list use index 0 only. A level without a separator is joined with a space, and extra separators are ignored.
- The same separators are used for the problem and the answer, so the result is not always what `DatasetWriter` wrote to the text file, where a flat list always gets `" | "`. For a nested problem with a flat answer, `["||", "|"]` turns the problem `[["1", "2"], ["3", "4"]]` into `1 | 2 || 3 | 4` as in the text file, but the answer `["5", "6"]` into `5 || 6`, where the text file has `5 | 6`.
- Empty lists become `""` and a one-element list is its element. Only lists are joined; a tuple in pickle data is converted with `str()` as before. Without `list_separators`, every level is joined with a space as before.
- It applies only to the default JSONL and pickle loaders. Text files are used as they are, and with `dataset_load_preprocessor` (e.g. `UserCallableLoadPreprocessor`, which receives the lists themselves) the setting has no effect; `IOPipeline` warns in both cases.

Every separator that ends up in the text must be in the vocabulary. List them in `vocab.misc` of `lexer.yaml`, including `"||"` when you use it (it is read as one token, and is not in the vocabulary by default):

```yaml
vocab:
  range:
    numbers: ["", 0, 9]
  misc: ["|", "||"]
number_policy:
  digit_group: 1
```

With this lexer and `list_separators: ["||", "|"]`, `[["31600", "54662"], ["1", "2"]]` is tokenized as `3 1 6 0 0 | 5 4 6 6 2 || 1 | 2`, so the element boundaries survive tokenization.

::: calt.io.preprocessor.load_preprocessor.UserCallableLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "UserCallableLoadPreprocessor"


::: calt.io.preprocessor.load_preprocessors.chain.ChainLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "ChainLoadPreprocessor"

::: calt.io.preprocessor.load_preprocessors.expanded_form.ExpandedFormLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "ExpandedFormLoadPreprocessor"

::: calt.io.preprocessor.load_preprocessors.text_to_sage.TextToSageLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "TextToSageLoadPreprocessor"

::: calt.io.preprocessor.load_preprocessors.reversed_order.ReversedOrderLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "ReversedOrderLoadPreprocessor"

::: calt.io.preprocessor.load_preprocessors.last_element.LastElementLoadPreprocessor
    options:
      members: ["__init__"]
      heading: "LastElementLoadPreprocessor"


