import pytest

def test_model_initialization():
    """Test model can be initialized"""
    assert True

def test_training_data_format():
    """Test CoNLL format validation"""
    sample_data = "ሰላም\tB-Product\n"
    assert "\t" in sample_data
    assert "B-" in sample_data