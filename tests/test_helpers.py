import pytest
from unittest.mock import MagicMock
import os
import src.helpers

def test_hf_login_failure(monkeypatch):
    # Mock the Hugging Face login to raise an exception
    mock_login = MagicMock(side_effect=Exception("Simulated login failure"))
    monkeypatch.setattr(src.helpers, "login", mock_login)

    # Mock environment variable and dotenv loading
    monkeypatch.setattr(os, "getenv", lambda key: "fake_token")
    monkeypatch.setattr(src.helpers, "load_dotenv", lambda: None)

    # Create a mock logger
    mock_logger = MagicMock()

    # Call the function and expect it to raise the exception
    try:
        src.helpers.hf_login(mock_logger)
    except Exception as e:
        # Assert that logger.exception was called with the correct message
        mock_logger.exception.assert_called_with(
            "There was a problem with authentication to HuggingFace\nError: Simulated login failure"
        )
    else:
        assert False, "Expected an exception but none was raised"

@pytest.fixture
def mock_accepted_values(monkeypatch):
    monkeypatch.setattr(
        "config.pipeline_config.accepted_values",
        {
            "er_status": ['a','b','c'],
            "er_score": ['2','3','4'],
            "pr_status": ['a','b','c'],
            "pr_score": ['2','3','4'],
            "her2_status": ['a','b','c']
        }
    )
    yield

@pytest.mark.parametrize(
    "input_dict, expected_status",
    [
        ({"er_score":"2","er_status":"a"},"valid"),
        ({"er_score":"99","er_status":"b"},"partial"),
        ({"er_score":"99","er_status":"to follow"},"invalid")
    ]
)
def test_check_valid_values(input_dict, expected_status,mock_accepted_values):
    num_cols = ["er_score"]
    stat_cols = ["er_status"]
    accepted_values = {
            "er_status": ['a','b','c'],
            "er_score": ['2','3','4']
        }
    result = src.helpers.check_valid_values(input_dict, accepted_values)
    assert result == expected_status