import pytest
from src.prompt_builder import PromptBuilder, PromptTemplateError
from unittest.mock import MagicMock

@pytest.fixture
def mock_boto3_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_prompt.return_value = {"variants":[{"templateConfiguration":{"text":{"text":"This is a fake prompt with {document} and {accepted_values}"}}}]}
    def _mock_boto3_client(service_name, *args, **kwargs):
        assert service_name == 'bedrock-agent'
        return mock_client

    import boto3
    monkeypatch.setattr(boto3, "client", _mock_boto3_client)
    return mock_client

class TestPromptBuilder():
    @pytest.mark.parametrize(
        "prompt_layout, kwargs, input_text",
        [
            ("This is the base instruction\nFields and their accepted values:\n{accepted_values}\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document.",
             {"accepted_values": {"ER Status":["P","N","B","U","X"]}}, 
             "This is a path report"),
            
            ("This is the base instruction\nFields and their accepted values:\n{accepted_values}\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document.\n Here are some examples:\n{examples}", 
             {"accepted_values": {"ER Status":["P","N","B","U","X"]}, "examples": [{"text":"example report","entities":{"ER Status":"P"}}]},
             "This is a path report")
        ],
    )
    def test_prompt_builder_output(self,prompt_layout, kwargs, input_text):
        model_id = "mistral.for.example"
        prompt_builder = PromptBuilder(model_id, prompt_layout, **kwargs)
        output = prompt_builder.build(input_text, 1)
        assert isinstance(output, str)

    @pytest.mark.parametrize(
        "prompt_layout, kwargs, input_text, expected_error",
        [
            ("This is the base instruction\nFields and their accepted values:\n{accepted_values}\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document. Here are some examples: {examples}",
             {"accepted_values": {"ER Status":["P","N","B","U","X"]}}, 
             "This is a path report","The placeholder '{examples}' in the supplied prompt is missing it's corresponding argument"),
            
            ("This is the base instruction\nFields and their accepted values:\n{accepted_values}\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document.", 
             {"accepted_values": {"ER Status":["P","N","B","U","X"]}, "examples": [{"text":"example report","entities":{"ER Status":"P"}}]},
             "This is a path report","The argument 'examples' was not found in the supplied prompt")
        ],
    )
    def test_prompt_builder_errors(self,prompt_layout, kwargs, input_text, expected_error):
        model_id = "mistral.for.example"
        with pytest.raises(PromptTemplateError) as err:
            PromptBuilder(model_id, prompt_layout, **kwargs)
        assert  expected_error in str(err.value)

    def test_prompt_format_error(self):
        model_id = "unsupported.model.id"
        prompt_layout = "This is the base instruction\n Format the output as a json with a key for each field where the value of the key is the corresponding entity from the document."
        prompter = PromptBuilder(model_id, prompt_layout)
        with pytest.raises(ValueError):
            prompter.build(input_text="fake doc", doc_id=0)

    def test_prompt_builder_prompt_management(self, mock_boto3_client):        
        model_id = "mistral.for.example"
        prompt_builder = PromptBuilder(model_id, prompt_id = 'FAKE_ID', prompt_version = 2, accepted_values = {"ER Status":["P","N","B","U","X"]})
        output = prompt_builder.build(input_text = 'this is a path report', doc_id = 1)
        assert isinstance(output, str)
        assert 'this is a path report' in output
        assert 'ER Status: P, N, B, U, X' in output

    @pytest.mark.parametrize(
        "prompt_layout, prompt_id, prompt_version, kwargs, expected_error",
        [
            ("fake_layout","fake_prompt_id",1,{"accepted_values": {"ER Status":["P","N","B","U","X"]}},"Exactly one of 'prompt_layout' or 'prompt_id' must be provided."),# both prompt layout and prompt_id supplied
            (None,None,1,{"accepted_values": {"ER Status":["P","N","B","U","X"]}},"Exactly one of 'prompt_layout' or 'prompt_id' must be provided."),# neither prompt_layout or prompt_id supplied
            (None,"fake_prompt_id",None,{"accepted_values": {"ER Status":["P","N","B","U","X"]}},"'prompt_version' must be provided when using 'prompt_id'."),# prompt_id supplied without version
        ]
    )
    def test_prompt_builder_prompt_management_errors(self, prompt_layout, prompt_id, prompt_version, kwargs, expected_error):
        model_id = "mistral.for.example"
        with pytest.raises(ValueError) as err:
            PromptBuilder(model_id, prompt_layout=prompt_layout, prompt_id=prompt_id, prompt_version=prompt_version, **kwargs)
        assert expected_error == str(err.value)
        