import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.helpers import load_config_from_yaml
from src.extractor_pipeline import ExtractorPipeline
from src.model_request import ModelRequest
from src.prompt_builder import PromptBuilder
from src.text_preprocessor import TextPreprocessor
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
def callable_preprocessor():
    preprocesser = TextPreprocessor()
    yield preprocesser

    del preprocesser

    
@pytest.fixture
def mock_preprocessor():
    preprocessor = MagicMock()
    preprocessor.process.return_value = "preprocessed text"
    return preprocessor


@pytest.fixture
def callable_prompt_builder():
    model_id = "fake.model.id"
    prompt_layout = "This is the base instruction\nFields and their accepted values:\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document. Document: {document}"
    prompt_builder = PromptBuilder(model_id, prompt_layout)
    yield prompt_builder
    
    del prompt_builder

    
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
def mock_get_tokenizer():
    def fake_tokenizer(x):
        return {'input_ids': [123,456,789,321,654,987], 'attention_mask': [1, 1, 1, 1, 1]}
    with patch.object(ExtractorPipeline, "_get_tokenizer", return_value=fake_tokenizer) as mock_method:
        yield mock_method

@pytest.fixture
def mock_record_cost():
    with patch.object(ExtractorPipeline, "record_cost", return_value=None) as mock_method:
        yield mock_method

@pytest.fixture
def mock_load_df_from_s3():
    def fake_load(bucket, key):
        return pd.DataFrame({
                "timestamp": [pd.Timestamp.now(), pd.Timestamp.now()],# this col is ignored in the assert
                "model_id": ['model1','model2'], 
                "model_args": ['{"arg1": 100, "arg2":200}','{"arg1": 100, "arg2":200}'], 
                "num_records":[5,10],
                "total_cost": [0.00345, 0.000678]
            })
    with patch("src.extractor_pipeline.load_dataframe_from_s3", side_effect=fake_load) as mock:
        yield mock

@pytest.fixture
def mock_save_df_to_s3():
    def fake_save(df, bucket, key):
        return df

    with patch("src.extractor_pipeline.save_dataframe_to_s3", side_effect=fake_save) as mock:
        yield mock

@pytest.fixture
def pipeline(mock_config, mock_preprocessor, mock_model_request, mock_model_config, standard_accepted_values):
    return ExtractorPipeline(
        config_file_path="dummy_path.yaml",
        preprocessor=mock_preprocessor,
        model_request=mock_model_request,
        valid_structure = vconf.ValidSchema,
        accepted_values = standard_accepted_values
    )
      
class TestExtractorPipeline:


    def test_initialisation(self, pipeline):
        """
        Tests the initialisation of the ExtractorPipeline to ensure
        that 'id_col' and 'free_text_col' are correctly set based on the config.
        """
        assert pipeline.id_col == id_col
        assert pipeline.free_text_col == freetext_col

    def test_model_config_error(self, callable_preprocessor, callable_prompt_builder, standard_accepted_values):
        model_id = "unsupported.model"
        model_args = {"arg1":200, "arg2":0, "arg3":0.9}
        expected = "'Check model_id, no config exists for the supplied model'"
        model_requester = ModelRequest(model_id,
                                       model_args,
                                       callable_prompt_builder)
        with pytest.raises(KeyError) as err:
            ExtractorPipeline("./config/local.yaml",
                              callable_preprocessor,
                              model_requester,
                              valid_structure = vconf.ValidSchema,
                              accepted_values = standard_accepted_values)

        assert expected == str(err.value)

    def test_process_row(self, pipeline, mock_preprocessor, mock_model_request):
        """
        Tests the `process_row` method of the ExtractorPipeline.

        It verifies that:
        - The method correctly processes a single row of data.
        - The preprocessor and model_request mocks are called exactly once.
        - The output dictionary contains the expected keys and values.
        """
        row = pd.Series({id_col: "123", freetext_col: "raw input text"})
        result = pipeline.process_row(row)

        assert result[id_col] == "123"
        assert result[freetext_col] == "raw input text"
        assert result[f"preprocessed_{freetext_col}"] == "preprocessed text"
        assert result["model_output"] == 'here is your json: {"key1":"val1","key2":0}'
        mock_preprocessor.process.assert_called_once()
        mock_model_request.call.assert_called_once()
        
    def test_run(self, pipeline, mock_record_cost):
        """
        Tests the `run` method of the ExtractorPipeline.

        It verifies that:
        - The method processes a DataFrame of input data.
        - The output is a pandas DataFrame.
        - The output DataFrame has the expected number of rows.
        - The output DataFrame contains all the necessary columns after processing.
        """
        df = pd.DataFrame([
            {id_col: "1", freetext_col: "test one"},
            {id_col: "2", freetext_col: "test two"},
        ])
        output_df = pipeline.run(df,
                                 estimate_cost=False,
                                 calculate_cost=False)
        
        assert isinstance(output_df, pd.DataFrame)
        assert len(output_df) == 2
        assert set(output_df.columns) >= {
            id_col,
            freetext_col,
            f"preprocessed_{freetext_col}",
            "model_output"
        }

    @pytest.mark.parametrize(
        "model_id",
        [
            ("mistral.mistral-7b-instruct-v0:2"),
            ("meta.llama3-8b-instruct-v1:0")
        ],
    )
    def test_estimate_cost_output_types(self, model_id, callable_prompt_builder, mock_get_tokenizer):
        model_args = {"arg1":200, "arg2":0, "arg3":0.9}
        doc_df = pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"]})
        prompt_layout = "Example prompt. Document: {document}"
        prompter = PromptBuilder(model_id, prompt_layout)
        model_requester = ModelRequest(model_id, model_args, prompter)
        extractor = ExtractorPipeline("./config/local.yaml",
                                      callable_preprocessor,
                                      model_requester,
                                      valid_structure = vconf.ValidSchema,
                                      accepted_values = standard_accepted_values)
        output = extractor.estimate_cost(doc_df, text_column=freetext_col)
        
        assert isinstance(output, pd.DataFrame)
        assert all(col in output.columns for col in ["est_input_cost","est_output_cost"])
        assert all(isinstance(x, float) for x in output["est_input_cost"].tolist())
        assert all(isinstance(x, float) for x in output["est_output_cost"].tolist())

    def test_estimate_cost_no_support(self, callable_prompt_builder, mock_get_tokenizer):
        """
        Check this function does nothing when cost estimation is not supported for a model
        i.e. when the required dictionary key is missing in the model config
        """
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        model_args = {"arg1":200, "arg2":0, "arg3":0.9}
        doc_df = pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"]})
        
        model_requester = ModelRequest(model_id, model_args, callable_prompt_builder)
        extractor = ExtractorPipeline("./config/local.yaml",
                                      callable_preprocessor,
                                      model_requester,
                                      valid_structure = vconf.ValidSchema,
                                      accepted_values = standard_accepted_values)
        output = extractor.estimate_cost(doc_df, text_column=freetext_col)
        
        pd.testing.assert_frame_equal(doc_df, output)
        
    @pytest.mark.parametrize(
        "model_id",
        [
            ("mistral.mistral-7b-instruct-v0:2"),
            ("meta.llama3-8b-instruct-v1:0")
        ],
    )
    def test_calculate_cost_output_types(self, model_id, callable_preprocessor, callable_prompt_builder, standard_accepted_values):
        model_args = {"arg1":200, "arg2":0, "arg3":0.9}
        doc_df = pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"],"input_tokens":[20,25],"output_tokens":[10,15]})
        
        model_requester = ModelRequest(model_id, model_args, callable_prompt_builder)
        extractor = ExtractorPipeline("./config/local.yaml",
                                      callable_preprocessor,
                                      model_requester,
                                      valid_structure = vconf.ValidSchema,
                                      accepted_values = standard_accepted_values)
        output = extractor.calculate_cost(doc_df)
        
        assert isinstance(output, pd.DataFrame)
        assert all(col in output.columns for col in ["actual_input_cost","actual_output_cost"])
        assert all(isinstance(x, float) for x in output["actual_input_cost"].tolist())
        assert all(isinstance(x, float) for x in output["actual_output_cost"].tolist())

    @pytest.mark.parametrize(
        "doc_df",
        [
            (pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"],"output_tokens":[10,15]})),
            (pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"],"input_tokens":[10,15]})),
            (pd.DataFrame({id_col:[1,2], freetext_col:["report1","report2"]}))
        ],
    )
    def test_calc_cost_col_error(self, doc_df, callable_preprocessor, callable_prompt_builder, standard_accepted_values):
        model_id = "mistral.mistral-7b-instruct-v0:2"
        model_args = {"arg1":200, "arg2":0, "arg3":0.9}
        expected = "The provided dataframe does not contain the required columns for cost calculation"
        
        model_requester = ModelRequest(model_id,
                                       model_args,
                                       callable_prompt_builder)
        
        extractor = ExtractorPipeline("./config/local.yaml",
                                      callable_preprocessor,
                                      model_requester,
                                      valid_structure = vconf.ValidSchema,
                                      accepted_values = standard_accepted_values)
        with pytest.raises(Exception) as err:
            extractor.calculate_cost(doc_df)
            
        assert expected == str(err.value)

    def test_record_cost(self, pipeline, mock_load_df_from_s3, mock_save_df_to_s3):
        fake_pipeline_result = pd.DataFrame({
            "actual_input_cost": [0.0002, 0.0003],
            "actual_output_cost": [0.000015, 0.000015],
        })
        expected_df = pd.DataFrame({
            "model_id": ['mistral.for.example','model1','model2'], 
            "model_args": ['{"arg1": "value1"}','{"arg1": 100, "arg2":200}','{"arg1": 100, "arg2":200}'], 
            "num_records":[2,5,10],
            "total_cost": [0.00053,0.00345, 0.000678]
        })
        pipeline.record_cost(fake_pipeline_result)
        cost_df = mock_save_df_to_s3.call_args[0][0]# Intercept the df from the mock
        
        mock_load_df_from_s3.assert_called_once()
        mock_save_df_to_s3.assert_called_once()
        assert "timestamp" in cost_df.columns# check this col exists since we remove it later
        assert pd.api.types.is_datetime64_any_dtype(cost_df["timestamp"])
        pd.testing.assert_frame_equal(
            cost_df.drop(columns=["timestamp"]).reset_index(drop=True),
            expected_df
        )
    
    @pytest.mark.parametrize(
        "model_text, doc_id, expected_output",
        [
            ('Here is your JSON in markdown ```json\n{"key1":"value1","key2":0}\n```',0,{"key1":"value1","key2":0}),
            ('Here is your JSON \n{"key1":"value1","key2":0}',0,{"key1":"value1","key2":0}),
            ('Here are some brackets but no JSON \n{no content in here}\n```',0,{"error":"error when loading from regex"}),
            ('No JSON to find in here!',0,{"error":"no json found"}),
        ],
    )
    def test_parse_output(self, model_text, doc_id, expected_output, pipeline):
        output = pipeline.parse_output(model_text, doc_id)
        assert output == expected_output


class TestCMEValidation():
    """
    This test is specific to the Clinical Measurement Extraction project for extracting
    ER/PR score and ER/PR/HER2 status from breast cancer pathology reports. If using this
    code for a different usecase, this test and the function being tested may not be relevant
    or may need adjusting accordingly.
    """
    @pytest.mark.parametrize(
             "parsed_json, expected_output",
            [
                ({"er_status":"a",
                  "er_score":"4",
                  "pr_status":"c",
                  "pr_score":"2",
                  "her2_status":"b"},
                 ({"er_status":"a",
                  "er_score":"4",
                  "pr_status":"c",
                  "pr_score":"2",
                  "her2_status":"b"},"valid")),# json exactly right
                ({"er_status":"a",
                  "er_score":"3",
                  "pr_score":"4",
                  "her2_status":"b"},
                 ({"er_status":"a",
                   "er_score":"3",
                   "pr_status":"key_missing",
                   "pr_score":"4",
                   "her2_status":"b"},"partial")),# missing key
                ({"er_status":"c",
                  "er_score":"2",
                  "pr_status":"a",
                  "pr_score":"2",
                  "her2_status":"a",
                  "extra_key":"extra"},
                 ({"er_status":"c",
                   "er_score":"2",
                   "pr_status":"a",
                   "pr_score":"2",
                   "her2_status":"a"},"valid")),# extra key
                ({"er_status":"a",
                  "er_score":"2",
                  "pr_status":"a",
                  "pr_score":3,
                  "her2_status":"a"},
                 ({"er_status":"a",
                   "er_score":"2",
                   "pr_status":"a",
                   "pr_score":"3",
                   "her2_status":"a"},"valid")),# incorrect but fixable type
                ({"er_status":"b",
                  "er_score":"zero",
                  "pr_status":"a",
                  "pr_score":"3",
                  "her2_status":"c"},
                 ({"er_status":"b",
                   "er_score":"zero",
                   "pr_status":"a",
                   "pr_score":"3",
                   "her2_status":"c"},"partial")),# incorrect unfixable type
                ({"er_status":"b",
                  "er_score":"10",
                  "pr_status":"something else",
                  "pr_score":"3",
                  "her2_status":"c"},
                 ({"er_status":"b",
                   "er_score":"10",
                   "pr_status":"something else",
                   "pr_score":"3",
                   "her2_status":"c"},"partial")),# correct structure and type but value out of range
                ({"er_status_a":"x",
                  "er_score_a":"10",
                  "pr_status_a":"y",
                  "pr_score_a":"20",
                  "her2_status_a":"wrong"},
                 ({"er_status":"key_missing",
                   "er_score":"key_missing",
                   "pr_status":"key_missing",
                   "pr_score":"key_missing",
                   "her2_status":"key_missing"},"invalid")),#everything wrong/missing
                ({"error":"something wrong"},
                 (None,"validation_failed")),# json parsing went wrong
            ],
        )
    def test_validate_update_json(self,parsed_json, expected_output, pipeline):
        doc_id = 0
        output = pipeline.validate_update_json(parsed_json, doc_id)
        assert output == expected_output
        