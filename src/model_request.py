import json
import boto3
from dotenv import load_dotenv
import os
import logging

from src.prompt_builder import PromptBuilder

load_dotenv()

class ModelRequest():
    """
    This class and it's methods can be used to send a request to AWS Bedrock models. Each model has it's own functions for calling the API.

    Args:
        model_id (str): the Bedrock model ID
        model_args (dict): model parameters for the model you are querying (e.g. max_tokens, temperature)
        prompt (PromptBuilder): instance of the PromptBuilder class

    """
    def __init__(self, model_id, model_args: dict, prompt: PromptBuilder):
        self.model_id = model_id
        self.model_args = model_args
        self.prompt = prompt
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.region = os.getenv("AWS_REGION")

        self.client = boto3.client("bedrock-runtime", region_name=self.region)


    def create_mistral_request_body(self, doc: str, doc_id:int, args: dict) -> dict:
        self.logger.info(f"Document id: {doc_id} | Creating Mistral's request body")
        
        prompt_for_body = self.prompt.build(doc, doc_id)
        prompt_for_body = {"prompt": prompt_for_body}
        
        body_json = {**prompt_for_body, **args}
        
        return json.dumps(body_json)


    def create_llama_request_body(self, doc: str, doc_id:int, args: dict) -> dict:
        self.logger.info(f"Document id: {doc_id} | Creating Llama's request body")
        prompt_for_body = self.prompt.build(doc, doc_id)
        prompt_for_body = {"prompt": prompt_for_body}
        
        body_json = {**prompt_for_body, **args}
        
        return json.dumps(body_json)

    def create_claude_request_body(self, doc:str, doc_id:int, args: dict) -> dict:
        self.logger.info(f"Document id: {doc_id} | Creating Claude's request body")
        system_prompt = self.prompt.system_prompt
        prompt_for_body = self.prompt.build(doc, doc_id)
        prompt_for_body = {"messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_for_body
                    }
                ]
            }
        ]}
        if system_prompt:
            prompt_for_body = {**{"system":system_prompt},**prompt_for_body}
            
        body_json = {**prompt_for_body, **args}
        
        return json.dumps(body_json)
        
    def get_model_id(self):
       return self.model_id

    def get_model_args(self):
       return self.model_args
    
    def call(self, doc: str, doc_id:int) -> dict:
        """
        Calls a bedrock model and returns the response body from the API

        Args:
            doc (str): The document from which the extraction will be done
            doc_id (int): The id associated to the document.
        Returns:
            dict: the response body from the model API
        """

        self.logger.info(f"Document id: {doc_id} | Invoking model request call method")
        
        if "mistral" in self.model_id:
            body_json = self.create_mistral_request_body(doc, doc_id, self.model_args)
        elif "llama" in self.model_id:
            body_json = self.create_llama_request_body(doc, doc_id, self.model_args)
        elif "claude" in self.model_id:
            # untested at this point as we don't have access to claude yet
            body_json = self.create_claude_request_body(doc, doc_id, self.model_args)
        
        request_json = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": body_json
        }
        try:
            response = self.client.invoke_model(**request_json)
            self.logger.info(f"Document id: {doc_id} | Model invoked successfully")
        except Exception as e:
            self.logger.exception(
                f"Document id: {doc_id} | Error invoking model with error {e}"
            )
            raise

        model_response = json.loads(response["body"].read())
        response_meta_data = response["ResponseMetadata"]

        return model_response, response_meta_data

             