import sys
import os
import warnings
import pytest

# Add project root to Python path
# This fixes ALL import errors in tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Suppress Pydantic v2 deprecation warnings from llama_index
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*extra keyword arguments on.*Field.*")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")

def pytest_configure(config):
    """Configure pytest to ignore Pydantic deprecation warnings."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pydantic.PydanticDeprecatedSince20"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*extra keyword arguments.*:DeprecationWarning"
    )