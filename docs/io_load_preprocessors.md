# Load preprocessors

Load preprocessors run once at load time (before the lexer) to convert raw file content—a text line, JSONL object, or pickle sample—into `(input_text, target_text)` pairs. You can use the library-provided implementations or supply your own and pass them to [IOPipeline](io_pipeline.md) via configuration.


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


