import pytest

def test_basic():
    """Basic test to ensure CI pipeline works"""
    assert True

def test_string_processing():
    """Test basic string operations for Amharic text"""
    text = "ሰላም"
    assert len(text) > 0
    assert isinstance(text, str)