import boto3
from typing import List
import json
import string
from dotenv import load_dotenv
import os
import logging

class PromptTemplateError(Exception):
    pass

class PromptBuilder():
    """
    This class creates a prompt using the initial base prompt containing optional parameters and kwargs corresponding to each parameter in the base prompt.
    The prompt can be built from an explicit prompt layout passed in as a string or you can pull a version of a prompt from AWS Prompt Management by supplying 
    the id and version number. Exactly one of prompt_layout or prompt_id must be supplied. The prompt_version must be supplied if using prompt_id.

    Each placeholder in the prompt (except {document}) must have a corresponding kwarg supplied.

    Args:
        model_id (str): model id to build the prompt for
        prompt_layout (str): the base prompt, including any placeholder parameters
        prompt_id (str): the id of the prompt in AWS Prompt Management, usually the last few characters of the ARN
        system_prompt (str): optional system prompt to complement main prompt
        prompt_id (str): prompt id from AWS Prompt Management
        prompt_version (int): the version of the prompt you want to use
        kwargs: optional arguments that should correspond to a placeholder parameter in the prompt
    """
    def __init__(self, model_id: str, prompt_layout: str = None, system_prompt: str = None, prompt_id: str = None, prompt_version: int = None, **kwargs):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_id = model_id
        self.prompt_layout = prompt_layout
        self.system_prompt = system_prompt
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        
        load_dotenv()
        self.region = os.getenv("AWS_REGION")
        
        self._validate_class_arguments()                            

        if prompt_id:
            self._load_from_prompt_management()
            
        self.prompt_kwargs = kwargs
        self.accepted_values = kwargs.get("accepted_values",None)
        self.examples = kwargs.get("examples",None)
        
        self._validate_prompt(self.prompt_layout, self.prompt_kwargs)

    def _load_from_prompt_management(self):
        client = boto3.client("bedrock-agent", region_name=self.region)
        response = client.get_prompt(promptIdentifier = self.prompt_id, promptVersion = str(self.prompt_version))
        self.prompt_layout = response["variants"][0]["templateConfiguration"]["text"]["text"]
        self.logger.info(f"Prompt {self.prompt_id} version {self.prompt_version} successfully loaded")

        
    def _validate_class_arguments(self):
        if (self.prompt_layout is None) == (self.prompt_id is None):
            message = "Exactly one of 'prompt_layout' or 'prompt_id' must be provided."
            self.logger.error(message)
            raise ValueError(message)
        if self.prompt_id and not self.prompt_version:
            message = "'prompt_version' must be provided when using 'prompt_id'."
            self.logger.error(message)
            raise ValueError(message)

    def _validate_prompt(self, prompt: str, args: dict):
        self.logger.info("Validating prompt template")
        for arg in args.keys():
            if "{"+arg+"}" not in prompt:
                message = f"The argument '{arg}' was not found in the supplied prompt"
                self.logger.error(message)
                raise PromptTemplateError(message)
                
        formatter = string.Formatter()
        expected_params = [field_name for _, field_name, _, _ in formatter.parse(prompt) if field_name]
        for expected in expected_params:
            if expected not in args.keys() and expected != "document":
                message = "The placeholder '{" + expected + "}' in the supplied prompt is missing it's corresponding argument"
                self.logger.error(message)
                raise PromptTemplateError(message)
            

    def _create_accepted_vals_str(self, fields: dict) -> str:
        self.logger.info("Creating accepted values to be added into prompt")
        accepted_values_str = "\n".join(
                [f"{key}: {', '.join(map(str, values))}" for key, values in fields.items()]
            )

        return accepted_values_str

    def _create_example_str(self, examples: List[dict]) -> str:
        self.logger.info("Creating examples to be added into Prompt")
        examples_str = "\n\n".join([
                f"Example:\nText: {ex['text']}\nEntities: {json.dumps(ex['entities'])}"
                for ex in examples
            ])

        return examples_str

    def _format_prompt(self):
        if "mistral" in self.model_id:
            formatted_prompt = f"<s>[INST] {self.prompt_layout} [/INST]"
        elif "llama" in self.model_id and self.system_prompt:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{self.prompt_layout}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        elif "llama" in self.model_id and not self.system_prompt:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{self.prompt_layout}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        elif any(model in self.model_id for model in ["claude","openai"]):
            formatted_prompt = self.prompt_layout
        else:
            raise ValueError("The model_id supplied to PromptBuilder is not supported")

        return formatted_prompt
            

    def list_prompt_versions(self, prompt_id: str):
        """
        Use this function to list all the version numbers for the specified prompt_id.

        Args:
            prompt_id (str): the id of the prompt in AWS Prompt Management, usually the last few characters of the ARN
        Returns:
            list: the versions of the prompt
        """
        client = boto3.client("bedrock-agent", region_name=self.region)
        response = client.list_prompts(promptIdentifier = prompt_id, maxResults = 500)
        prompts = response["promptSummaries"]
        prompts_sorted = sorted(prompts, key=lambda d: d['createdAt'])
        versions = [(prompt.get("version"), prompt.get("description")) for prompt in prompts_sorted]
        return versions

    def save_prompt_version(self, 
                            prompt_id: str,
                            prompt_name: str,
                            version_num: int, 
                            prompt_description: str = None):
        """
        Use this function to save a new version of the prompt. Note that prompt_version is 
        required so that you have edit the parameter each time to avoid accidently saving
        the same prompt multiple times. The version_num provided is not necessarily the 
        version number that will be used to save the prompt.

        Args:
            prompt_id (str): the id of your AWS Prompt Management space
            prompt_name (str): the name of your AWS Prompt Management space
            version_num (int): This should be the next version number in the sequence
            prompt_description (str): optional description for this prompt version
        """
        version_list = [v[0] for v in self.list_prompt_versions(prompt_id)]
        if 'DRAFT' in version_list: version_list.remove('DRAFT')
        version_list = list(map(int,version_list))
        
        if version_num - max(version_list, default = 0) != 1:
            raise Exception(f"Version '{version_num}' is not the next in sequence, prompt not saved.")
        else:
            client = boto3.client("bedrock-agent", region_name=self.region)
            prompt_input = [{'name': prompt_name, 
                'templateConfiguration': {
                    'text': {
                        'inputVariables': [], 
                        'text': self.prompt_layout
                    }
                }, 
                  'templateType': 'TEXT'
                 }
                ]
            client.update_prompt(
                promptIdentifier = prompt_id,
                variants = prompt_input,
                name = prompt_name
            )# updates the draft version

            response = client.create_prompt_version(
                description = prompt_description,
                promptIdentifier = prompt_id
            )# saves draft as new version
            
            if response["ResponseMetadata"]["HTTPStatusCode"] == 201:
                version = response["version"]
                print(f"Successfully saved new prompt as version {version}")
            else:
                raise Exception(f"Prompt not updated. See reponse object:\n{response}")
    
    def build(self, input_text: str, doc_id:int) -> str:
        """
        This method builds the final prompt using the arguments supplied to the PromptBuilder class and inserts the input document into the prompt

        Args:
            input_text (str): The document from which the extraction will be done.
            doc_id (int): The document id associated with the text.
        Returns:
            str: the final prompt in string form
        """
        self.logger.info(f"Document id: {doc_id} | Building the prompt layout")
        if self.accepted_values:
            self.prompt_kwargs["accepted_values"] = self._create_accepted_vals_str(self.accepted_values)
        
        if self.examples:
            self.prompt_kwargs["examples"] = self._create_example_str(self.examples)

        formatted_prompt = self._format_prompt()
        prompt = formatted_prompt.format(**{**self.prompt_kwargs, **{"document": input_text}})

        return prompt
