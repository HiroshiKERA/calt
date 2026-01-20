import pytest

# These processors have been removed - skip all tests in this file
pytestmark = pytest.mark.skip(reason="Old preprocessors have been removed, use UnifiedLexer instead")
