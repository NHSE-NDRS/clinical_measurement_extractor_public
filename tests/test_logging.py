import pytest
import glob
import logging
import pandas as pd
import os
import shutil
from unittest.mock import MagicMock, patch

from src.helpers import load_config_from_yaml
from src.extractor_pipeline import ExtractorPipeline
from src.custom_logging import setup_logging
import config.validation_config as vconf

# Define config file path
conf_file_path = "./config/local.yaml"
# Load config
conf = load_config_from_yaml(file_path=conf_file_path)
id_col = conf.get("ID_COL")
freetext_col = conf.get("FREETEXT_COL")


@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setattr(
        'src.extractor_pipeline.load_config_from_yaml',
        lambda file_path: {
            "ID_COL": id_col,
            "FREETEXT_COL": freetext_col
        }
    )

@pytest.fixture
def mock_model_config(monkeypatch):
    monkeypatch.setattr(
        "src.extractor_pipeline.conf.model_config_all",
        {"mistral.for.example": {"config_param": "value"}}
    )
    yield

@pytest.fixture
def mock_preprocessor():
    preprocessor = MagicMock()
    preprocessor.process.return_value = "preprocessed text"
    return preprocessor

@pytest.fixture
def mock_model_request():
    model_request = MagicMock()
    
    model_response = {"outputs": [{"text":'Here is your json: {"key1":"val1","key2":0}'}]}

    model_response_meta = {
        "HTTPHeaders": {
            "x-amzn-bedrock-input-token-count": "20",
            "x-amzn-bedrock-output-token-count": "10"
        }
    }

    model_request.call.return_value = (model_response, model_response_meta)
    model_request.get_model_id.return_value = "mistral.for.example"
    model_request.get_model_args.return_value = {"arg1": "value1"}
    
    return model_request

@pytest.fixture
def log_dir_cleanup():
    log_dir = "test_logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    yield
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

@pytest.fixture
def mock_record_cost():
    with patch.object(ExtractorPipeline, "record_cost", return_value=None) as mock_method:
        yield mock_method

@pytest.fixture
def standard_accepted_values():
    values = {
            "er_status": ['a','b','c'],
            "er_score": ['2','3','4'],
            "pr_status": ['a','b','c'],
            "pr_score": ['2','3','4'],
            "her2_status": ['a','b','c']
        }
    return values

@pytest.fixture
def pipeline(mock_config, mock_preprocessor, mock_model_request, mock_model_config, standard_accepted_values):
    return ExtractorPipeline(
        config_file_path="dummy_path.yaml",
        preprocessor=mock_preprocessor,
        model_request=mock_model_request,
        valid_structure = vconf.ValidSchema, 
        accepted_values = standard_accepted_values
    )

class TestLogging:

    def test_logging_file_creation(self, log_dir_cleanup):
        """
        Tests that a log file is correctly created when file logging is enabled.

        It asserts that:
        - Exactly one log file is found in the specified log directory.
        - The content of the log file contains an expected initialisation message.
        """
        setup_logging(
            enable_console = False,
            enable_file = True,
            console_log_level = logging.INFO,
            log_dir = "test_logs")

        log_files = glob.glob("test_logs/pipeline_*.log")
        assert len(log_files) == 1

        with open(log_files[0], "r") as f:
            content = f.read()
        assert "Logging initialized" in content

    def test_logging_file_disabled_no_file(self, log_dir_cleanup):
        """
        Tests that no log file is created when file logging is disabled.

        It asserts that no log files are present in the log directory.
        """
        setup_logging(
            enable_console = False,
            enable_file = False,
            console_log_level = logging.INFO,
            log_dir = "test_logs")

        log_files = glob.glob("test_logs/pipeline_*.log")
        assert len(log_files) == 0

    def test_run_console_logging(self, pipeline, mock_record_cost, caplog, log_dir_cleanup):
        """
        Tests that console logs are generated when ExtractorPipeline.run() is called
        with console logging enabled.
        """
        # Set the logging level for the caplog fixture
        caplog.set_level(logging.INFO)

        df = pd.DataFrame([
            {id_col: "1", freetext_col: "test one"},
            {id_col: "2", freetext_col: "test two"},
        ])
        pipeline.run(df, estimate_cost=False, calculate_cost=False)

        # Check for the log message from the run method
        assert "Running the extractor pipeline" in caplog.text
        # Check for log messages from process_row calls within run
        assert "Document id: 1 | Processing row" in caplog.text
        assert "Document id: 2 | Processing row" in caplog.text

    def test_process_row_console_logging(self, pipeline, caplog, log_dir_cleanup):
        """
        Tests that console logs are generated when ExtractorPipeline.process_row() is called
        with console logging enabled.
        """
        caplog.set_level(logging.INFO)

        row = pd.Series({id_col: "456", freetext_col: "another input"})
        pipeline.process_row(row)

        # Check for the specific log message from the process_row method
        assert "Document id: 456 | Processing row" in caplog.text