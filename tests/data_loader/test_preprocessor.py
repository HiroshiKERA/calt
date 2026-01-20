import pytest

# These processors have been removed - skip all tests in this file
pytestmark = pytest.mark.skip(reason="PolynomialToInternalProcessor and IntegerToInternalProcessor have been removed")
