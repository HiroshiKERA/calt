import pytest

from calt.data_loader.utils.preprocessor import (
    AbstractPreprocessor,
    CoefficientPostfixProcessor,
    PolynomialToInternalProcessor,
    ProcessorChain,
)


class _SuffixPreprocessor(AbstractPreprocessor):
    def __init__(self, suffix: str) -> None:
        super().__init__(num_variables=0, max_degree=0, max_coeff=1)
        self.suffix = suffix

    def encode(self, text: str) -> str:
        return f"{text}{self.suffix}"

    def decode(self, tokens: str) -> str:
        if self.suffix and tokens.endswith(self.suffix):
            return tokens[: -len(self.suffix)]
        return tokens


@pytest.mark.parametrize(
    ("raw_tokens", "expected"),
    [
        ("C123 E1 E2", "E1 E2 C123"),
        ("C1 E0", "E0 C1"),
        ("C1 E1 E0 C2 E0 E1", "E1 E0 C1 E0 E1 C2"),
        ("E0 E1", "E0 E1"),
        ("C5", "C5"),
        ("", ""),
        ("  ", "  "),
    ],
)
def test_coefficient_postfix_processor_encode(raw_tokens: str, expected: str):
    processor = CoefficientPostfixProcessor()
    assert processor.encode(raw_tokens) == expected


def test_coefficient_postfix_processor_decode():
    processor = CoefficientPostfixProcessor()
    assert processor.decode("E1 E2 C3") == "C3 E1 E2"


def test_processor_chain_applies_preprocessors_in_order():
    chain = ProcessorChain([_SuffixPreprocessor("-a"), _SuffixPreprocessor("-b")])
    assert chain.encode("foo") == "foo-a-b"
    assert chain.decode("foo-a-b") == "foo"


def test_processor_chain_accepts_polynomial_and_postfix():
    poly = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=100)
    postfix = CoefficientPostfixProcessor()
    chain = ProcessorChain([poly, postfix])

    encoded = chain.encode("3*x0*x1^2")
    assert encoded == "E1 E2 C3"
    decoded = chain.decode(encoded)
    assert decoded.replace(" ", "") == "3*x0*x1^2"
