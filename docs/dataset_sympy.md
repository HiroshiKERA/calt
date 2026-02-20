# SymPy backend

When using `backend="sympy"`, the following classes are used for generation and sampling. You can also use them directly without [DatasetPipeline](dataset_generator.md).

See [Dataset Generator (Overview)](dataset_generator.md) for the pipeline and `data.yaml` configuration.

<!-- ## DatasetGenerator -->

::: calt.dataset.sympy.dataset_generator.DatasetGenerator
    options:
      heading: "DatasetGenerator"

<!-- ## Sampling -->

::: calt.dataset.sympy.utils.polynomial_sampler.PolynomialSampler
    options:
      heading: "PolynomialSampler"