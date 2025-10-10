import pytest
from calt.data_loader.utils.preprocessor import (
    PolynomialToInternalProcessor,
    IntegerToInternalProcessor,
)


# Fixtures for processors
@pytest.fixture
def poly_processor():
    return PolynomialToInternalProcessor(num_variables=3, max_degree=5, max_coeff=10)


@pytest.fixture
def poly_processor_chunked():
    return PolynomialToInternalProcessor(
        num_variables=2, max_degree=5, max_coeff=100, digit_group_size=3
    )


@pytest.fixture
def int_processor():
    return PolynomialToInternalProcessor(num_variables=0, max_degree=0, max_coeff=9)


@pytest.fixture
def int_processor_chunked():
    return PolynomialToInternalProcessor(
        num_variables=0, max_degree=0, max_coeff=9, digit_group_size=3
    )


@pytest.fixture
def deprecated_int_processor():
    with pytest.deprecated_call():
        return IntegerToInternalProcessor(max_coeff=9)


# -- PolynomialToInternalProcessor Tests --

# Test cases for encode and decode identity
poly_test_cases_identity = [
    "3*x0^2*x1 - 5*x2 + 2",
    "x0*x1*x2",
    "-x0",
    "10",
    "x0^5",
    "2*x0-3",
    "0",
]


@pytest.mark.parametrize("poly_str", poly_test_cases_identity)
def test_polynomial_processor_identity(poly_processor, poly_str):
    """
    Test decode(encode(text)) == text
    """
    internal_rep = poly_processor.encode(poly_str)
    # The reconstructed string might have slightly different spacing or ordering
    # so we compare the "processed" version of both. This is a weaker check.
    # A stronger check would be to parse both and compare structured representations.
    reconstructed_poly = poly_processor.decode(internal_rep)

    # To handle cosmetic differences like "2*x0 - 3" vs "2*x0-3",
    # we can normalize by removing spaces.
    normalized_original = poly_str.replace(" ", "")
    normalized_reconstructed = reconstructed_poly.replace(" ", "")

    assert normalized_reconstructed == normalized_original


# Test cases for decode and encode identity
@pytest.mark.parametrize("poly_str", poly_test_cases_identity)
def test_polynomial_processor_internal_identity(poly_processor, poly_str):
    """
    Test encode(decode(tokens)) == tokens
    """
    internal_rep = poly_processor.encode(poly_str)
    if internal_rep != "[ERROR_PARSING]":
        reconstructed_internal = poly_processor.encode(
            poly_processor.decode(internal_rep)
        )
        assert reconstructed_internal == internal_rep


# Specific test cases for polynomials
poly_tests = {
    # Constants
    "constant_1": ("5", "C5 E0 E0 E0"),
    "constant_2": ("-3", "C-3 E0 E0 E0"),
    "constant_3": ("0", "C0 E0 E0 E0"),
    # Monomials
    "monomial_1": ("x0", "C1 E1 E0 E0"),
    "monomial_2": ("-x1^2", "C-1 E0 E2 E0"),
    "monomial_3": ("2*x2^3", "C2 E0 E0 E3"),
    # Polynomials
    "poly_1": ("x0 + x1", "C1 E1 E0 E0 C1 E0 E1 E0"),
    "poly_2": ("2*x0^2 - 3*x1", "C2 E2 E0 E0 C-3 E0 E1 E0"),
    "poly_3": ("x0*x1*x2 - 1", "C1 E1 E1 E1 C-1 E0 E0 E0"),
}


@pytest.mark.parametrize("name", poly_tests.keys())
def test_polynomial_processor_cases(poly_processor, name):
    poly_str, internal_str = poly_tests[name]
    # Test encode
    assert poly_processor.encode(poly_str) == internal_str
    # Test decode after removing spaces for comparison
    reconstructed_poly = poly_processor.decode(internal_str)
    assert reconstructed_poly.replace(" ", "") == poly_str.replace(" ", "")


def test_polynomial_processor_digit_grouping(poly_processor_chunked):
    encoded = poly_processor_chunked.encode("12345*x0*x1^2")
    assert encoded == "C12 C345 E1 E2"
    decoded = poly_processor_chunked.decode(encoded)
    assert decoded.replace(" ", "") == "12345*x0*x1^2"


def test_polynomial_processor_digit_grouping_negative(poly_processor_chunked):
    encoded = poly_processor_chunked.encode("-12345*x0")
    assert encoded == "C-12 C345 E1 E0"
    decoded = poly_processor_chunked.decode(encoded)
    assert decoded.replace(" ", "") == "-12345*x0"


def test_polynomial_processor_zero_coeff_chunked(poly_processor_chunked):
    assert poly_processor_chunked.encode("0") == "C0 E0 E0"


# -- IntegerToInternalProcessor Tests --

# Test cases for integer identity
integer_test_cases_identity = [
    "12345",
    "0",
    "987",
    "1|2|3",
    "123|456",
    "0|00|1",
    "-12",
    "-12|34",
    "1|-2|03",
]


@pytest.mark.parametrize("int_str", integer_test_cases_identity)
def test_integer_processor_identity(int_processor, int_str):
    """
    Test decode(encode(text)) == text
    """
    internal_rep = int_processor.encode(int_str)
    reconstructed_int = int_processor.decode(internal_rep)
    assert reconstructed_int == int_str


@pytest.mark.parametrize("int_str", integer_test_cases_identity)
def test_integer_processor_internal_identity(int_processor, int_str):
    """
    Test encode(decode(tokens)) == tokens
    """
    internal_rep = int_processor.encode(int_str)
    if internal_rep != "[ERROR_FORMAT]":
        reconstructed_internal = int_processor.encode(
            int_processor.decode(internal_rep)
        )
        assert reconstructed_internal == internal_rep


# Specific test cases for integers
integer_tests = {
    # Single numbers
    "single_1": ("123", "C123"),
    "single_2": ("90", "C90"),
    "single_3": ("7", "C7"),
    "negative_single": ("-100", "C-100"),
    # Multiple numbers with |
    "multi_1": ("1|23", "C1 [SEP] C23"),
    "multi_2": ("8|9|0", "C8 [SEP] C9 [SEP] C0"),
    "multi_3": ("100|200", "C100 [SEP] C200"),
    "negative_multi": ("-12|34", "C-12 [SEP] C34"),
    # Leading zeros
    "leading_zero_1": ("01", "C01"),
    "leading_zero_2": ("007", "C007"),
    "leading_zero_3": ("0|1", "C0 [SEP] C1"),
    "negative_leading_zero": ("-007", "C-007"),
}


@pytest.mark.parametrize("name", integer_tests.keys())
def test_integer_processor_cases(int_processor, name):
    int_str, internal_str = integer_tests[name]
    # Test encode
    assert int_processor.encode(int_str) == internal_str
    # Test encode
    assert int_processor.decode(internal_str) == int_str


def test_integer_processor_digit_grouping_single(int_processor_chunked):
    encoded = int_processor_chunked.encode("12345")
    assert encoded == "C12 C345"
    assert int_processor_chunked.decode(encoded) == "12345"


def test_integer_processor_digit_grouping_with_leading_zeros(int_processor_chunked):
    encoded = int_processor_chunked.encode("007")
    assert encoded == "C007"
    assert int_processor_chunked.decode(encoded) == "007"


def test_integer_processor_digit_grouping_multiple_parts(int_processor_chunked):
    encoded = int_processor_chunked.encode("12|3456")
    assert encoded == "C12 [SEP] C345 C6"
    assert int_processor_chunked.decode(encoded) == "12|3456"


def test_integer_processor_digit_grouping_negative(int_processor_chunked):
    encoded = int_processor_chunked.encode("-12345")
    assert encoded == "C-12 C345"
    assert int_processor_chunked.decode(encoded) == "-12345"


def test_integer_processor_invalid_input(int_processor):
    assert int_processor.encode("12|34x") == "[ERROR_FORMAT]"


def test_integer_processor_wrapper_still_works(deprecated_int_processor):
    encoded = deprecated_int_processor.encode("123")
    assert encoded == "C123"
    assert deprecated_int_processor.decode(encoded) == "123"


# Backward compatibility check example from prompt
def test_backward_compatibility_example():
    proc = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=5)
    internal = proc.encode("2*x0 - 3")
    assert proc.decode(internal).replace(" ", "") == "2*x0-3"
