import pytest
from calt.data_loader.utils.preprocessor import PolynomialToInternalProcessor, IntegerToInternalProcessor


# Fixtures for processors
@pytest.fixture
def poly_processor():
    return PolynomialToInternalProcessor(num_variables=3, max_degree=5, max_coeff=10)


@pytest.fixture
def int_processor():
    return IntegerToInternalProcessor(max_coeff=9)


# -- PolynomialToInternalProcessor Tests --

# Test cases for to_internal and to_original identity
poly_test_cases_identity = ["3*x0^2*x1 - 5*x2 + 2", "x0*x1*x2", "-x0", "10", "x0^5", "2*x0-3", "0"]


@pytest.mark.parametrize("poly_str", poly_test_cases_identity)
def test_polynomial_processor_identity(poly_processor, poly_str):
    """
    Test to_original(to_internal(text)) == text
    """
    internal_rep = poly_processor.to_internal(poly_str)
    # The reconstructed string might have slightly different spacing or ordering
    # so we compare the "processed" version of both. This is a weaker check.
    # A stronger check would be to parse both and compare structured representations.
    reconstructed_poly = poly_processor.to_original(internal_rep)

    # To handle cosmetic differences like "2*x0 - 3" vs "2*x0-3",
    # we can normalize by removing spaces.
    normalized_original = poly_str.replace(" ", "")
    normalized_reconstructed = reconstructed_poly.replace(" ", "")

    assert normalized_reconstructed == normalized_original


# Test cases for to_original and to_internal identity
@pytest.mark.parametrize("poly_str", poly_test_cases_identity)
def test_polynomial_processor_internal_identity(poly_processor, poly_str):
    """
    Test to_internal(to_original(tokens)) == tokens
    """
    internal_rep = poly_processor.to_internal(poly_str)
    if internal_rep != "[ERROR_PARSING]":
        reconstructed_internal = poly_processor.to_internal(poly_processor.to_original(internal_rep))
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
    # Test to_internal
    assert poly_processor.to_internal(poly_str) == internal_str
    # Test to_original after removing spaces for comparison
    reconstructed_poly = poly_processor.to_original(internal_str)
    assert reconstructed_poly.replace(" ", "") == poly_str.replace(" ", "")


# -- IntegerToInternalProcessor Tests --

# Test cases for integer identity
integer_test_cases_identity = ["12345", "0", "987", "1|2|3", "123|456", "0|00|1"]


@pytest.mark.parametrize("int_str", integer_test_cases_identity)
def test_integer_processor_identity(int_processor, int_str):
    """
    Test to_original(to_internal(text)) == text
    """
    internal_rep = int_processor.to_internal(int_str)
    reconstructed_int = int_processor.to_original(internal_rep)
    assert reconstructed_int == int_str


@pytest.mark.parametrize("int_str", integer_test_cases_identity)
def test_integer_processor_internal_identity(int_processor, int_str):
    """
    Test to_internal(to_original(tokens)) == tokens
    """
    internal_rep = int_processor.to_internal(int_str)
    if internal_rep != "[ERROR_FORMAT]":
        reconstructed_internal = int_processor.to_internal(int_processor.to_original(internal_rep))
        assert reconstructed_internal == internal_rep


# Specific test cases for integers
integer_tests = {
    # Single numbers
    "single_1": ("123", "C1 C2 C3"),
    "single_2": ("90", "C9 C0"),
    "single_3": ("7", "C7"),
    # Multiple numbers with |
    "multi_1": ("1|23", "C1 [SEP] C2 C3"),
    "multi_2": ("8|9|0", "C8 [SEP] C9 [SEP] C0"),
    "multi_3": ("100|200", "C1 C0 C0 [SEP] C2 C0 C0"),
    # Leading zeros
    "leading_zero_1": ("01", "C0 C1"),
    "leading_zero_2": ("007", "C0 C0 C7"),
    "leading_zero_3": ("0|1", "C0 [SEP] C1"),
}


@pytest.mark.parametrize("name", integer_tests.keys())
def test_integer_processor_cases(int_processor, name):
    int_str, internal_str = integer_tests[name]
    # Test to_internal
    assert int_processor.to_internal(int_str) == internal_str
    # Test to_original
    assert int_processor.to_original(internal_str) == int_str


# Backward compatibility check example from prompt
def test_backward_compatibility_example():
    proc = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=5)
    internal = proc.to_internal("2*x0 - 3")
    assert proc.to_original(internal).replace(" ", "") == "2*x0-3"
