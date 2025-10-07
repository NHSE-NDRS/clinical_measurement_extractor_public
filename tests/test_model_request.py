import pytest
from unittest.mock import MagicMock
from typing import Tuple
from botocore.response import StreamingBody
import io
import json

from src.prompt_builder import PromptBuilder
from src.model_request import ModelRequest

@pytest.fixture
def mock_invoke_model(monkeypatch):
    fake_model_response = json.dumps({"output":"This is the fake model response"}).encode()
    mock_botocore = StreamingBody(io.BytesIO(fake_model_response), len(fake_model_response))
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"ResponseMetadata":{'RequestId': 'abcdefg1234567', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Mon, 01 Jan 2000 12:34:56 GMT', 'content-type': 'application/json', 'content-length': '123', 'connection': 'keep-alive', 'x-amzn-requestid': 'hijklmn891011', 'x-amzn-bedrock-invocation-latency': '1234', 'x-amzn-bedrock-output-token-count': '150', 'x-amzn-bedrock-input-token-count': '150'}, 'RetryAttempts': 0},'contentType': 'application/json', "body": mock_botocore}
    def _mock_invoke_model(service_name, *args, **kwargs):
        assert service_name == 'bedrock-runtime'
        return mock_client

    import boto3
    monkeypatch.setattr(boto3, "client", _mock_invoke_model)
    return mock_client

@pytest.fixture
def callable_prompt_builder():
    model_id = "mistral.for.example"
    prompt_layout = "Example prompt. Document: {document}"
    prompt_builder = PromptBuilder(model_id, prompt_layout)
    yield prompt_builder
    
    del prompt_builder


class TestModelRequest():

    def test_call_outputs(self, callable_prompt_builder, mock_invoke_model):
        model_id = "mistral.for.example"
        model_args = {"arg1": "value1"}
        model_requester = ModelRequest(model_id, model_args, callable_prompt_builder)
        doc = "fake document"
        doc_id = 1
        output = model_requester.call(doc, doc_id)
        assert isinstance(output, Tuple)
        assert isinstance(output[0], dict)
        assert isinstance(output[1], dict)

    def test_get_model_args(self):
        model_id = "mistral.for.example"
        model_args = {"arg1": 0.5}
        model_requester = ModelRequest(model_id, model_args, callable_prompt_builder)
        output = model_requester.get_model_args()
        assert output == {"arg1": 0.5}

    def test_get_model_id(self):
        model_id = "mistral.for.example"
        model_args = {"arg1": 0.5}
        model_requester = ModelRequest(model_id, model_args, callable_prompt_builder)
        output = model_requester.get_model_id()
        assert output == "mistral.for.example"

    def test_create_claude_body(self):
        model_id = "claude.for.example"
        prompt_layout = "Example prompt.\nExample prompt. Document: {document}"
        prompt_builder = PromptBuilder(model_id, prompt_layout)
        
        model_args = {"arg1": 0.5}
        model_requester = ModelRequest(model_id, model_args, prompt_builder)
        
        doc = "fake document"
        doc_id = 1
        expected_output = '{"messages": [{"role": "user", "content": [{"type": "text", "text": "Example prompt.\\nExample prompt. Document: fake document"}]}], "arg1": 0.5}'
        output = model_requester.create_claude_request_body(doc, doc_id, model_args)
        
        assert output == expected_output

    @pytest.mark.parametrize(
        "system_prompt, expected",
        [
        (None, '{"prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\nExample prompt.\\nExample prompt. Document: fake document<|eot_id|><|start_header_id|>assistant<|end_header_id|>", "arg1": 0.5}'),
        ("You are an expert", '{"prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\nYou are an expert<|eot_id|><|start_header_id|>user<|end_header_id|>\\nExample prompt.\\nExample prompt. Document: fake document<|eot_id|><|start_header_id|>assistant<|end_header_id|>", "arg1": 0.5}')
    ])
    def test_create_llama_body(self, system_prompt, expected):
        model_id = "llama.for.example"
        prompt_layout = "Example prompt.\nExample prompt. Document: {document}"
        prompt_builder = PromptBuilder(model_id, prompt_layout, system_prompt)
        
        model_args = {"arg1": 0.5}
        model_requester = ModelRequest(model_id, model_args, prompt_builder)
        
        doc = "fake document"
        doc_id = 1
        output = model_requester.create_llama_request_body(doc, doc_id, model_args)
        
        assert output == expected

    def test_create_mistral_body(self):
        model_id = "mistral.for.example"
        prompt_layout = "Example prompt.\nExample prompt. Document: {document}"
        prompt_builder = PromptBuilder(model_id, prompt_layout)
        
        model_args = {"arg1": 0.5}
        model_requester = ModelRequest(model_id, model_args, prompt_builder)
        
        doc = "fake document"
        doc_id = 1
        expected_output = '{"prompt": "<s>[INST] Example prompt.\\nExample prompt. Document: fake document [/INST]", "arg1": 0.5}'
        output = model_requester.create_mistral_request_body(doc, doc_id, model_args)

        assert output == expected_output
    