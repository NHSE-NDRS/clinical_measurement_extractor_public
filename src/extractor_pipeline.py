import pandas as pd
import logging
import re
from typing import Tuple

from transformers import AutoTokenizer
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

from dotenv import load_dotenv
import json

from src.helpers import load_config_from_yaml, hf_login, check_valid_values
import config.pipeline_config as conf
import config.validation_config as vconf
from src.model_request import ModelRequest
from src.text_preprocessor import TextPreprocessor
from src.load_data import load_dataframe_from_s3, save_dataframe_to_s3
from src.evaluation_helpers import add_columns

load_dotenv()

class ExtractorPipeline:
    """
    This class and it's methods can be used to run an end-to-end pipeline of
    extracting out entities from free-text using LLMs and formating as a JSON output.

    Args:
        config_file_path (str): the path to where the file config lives.
        preprocessor (TextPreproccesor): This is a defined TextPreproccessor class that preprocesses the inputs.
        model_request (ModelRequest): This is a defined ModelRequest class that sends a request to the model to execute the extraction task.
        valid_structure (pydantic.BaseModel): a subclass of pydantics BaseModel used to check json is valid
        record_pipeline_cost (bool): whether to record the cost of the pipeline run to a csv in S3, default is True

    """

    def __init__(self,
                 config_file_path: str,
                 preprocessor: TextPreprocessor,
                 model_request: ModelRequest,
                 valid_structure: BaseModel,
                 accepted_values: dict,
                 record_pipeline_cost = True):
        
        self.preprocessor = preprocessor
        self.model_request = model_request
        self.valid_structure = valid_structure
        self.accepted_values = accepted_values
        config = load_config_from_yaml(file_path=config_file_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.record_pipeline_cost = record_pipeline_cost

        self.id_col = config.get("ID_COL")
        self.free_text_col = config.get("FREETEXT_COL")
        self.bucket_name = config.get("BUCKET_NAME")
        self.cost_logs = config.get("COST_LOGS")
        self.model_id = model_request.get_model_id()
        self.model_args = model_request.get_model_args()

        try:
            self.model_config = conf.model_config_all[self.model_id]
        except:
            error_msg = "Check model_id, no config exists for the supplied model"
            self.logger.exception(error_msg)
            raise KeyError(error_msg)

    def _calc_input_cost(self, row):
        x = (row/conf.cost_aggregation)*self.model_config["costs"]["input_cost_p1000"]
        return x
        
    def _calc_output_cost(self, row):
        x = (row/conf.cost_aggregation)*self.model_config["costs"]["output_cost_p1000"]
            
        return x

    def _get_tokenizer(self):
        hf_login(self.logger)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config["huggingface_path"])
        return tokenizer

    def _create_req_body(self, row, text_column):
        text = row[text_column]
        doc_id = row.get(self.id_col)
        if "mistral" in self.model_id:
            return json.loads(self.model_request.create_mistral_request_body(text, doc_id, self.model_args))["prompt"]
        elif "llama" in self.model_id:
            return json.loads(self.model_request.create_llama_request_body(text, doc_id, self.model_args))["prompt"]

    def _get_model_output_text(self, model_response):
        """
        This functions extracts the model response text from the model response json. 
        Different models output in different formats so this function handles for that.

        Note: Currently the model text is returned all lowercased to help validation steps later on,
        this might not be appropriate for every future usecase of this pipeline.
        """
        if "mistral" in self.model_id:
            model_text = model_response['outputs'][0]['text']
            model_text = model_text.replace(r"\_", "_")
        elif "llama" in self.model_id:
            model_text = model_response['generation']
        elif "claude" in self.model_id:
            model_text = model_response["content"][0]["text"]
        elif "openai" in self.model_id:
            model_text = model_response['choices'][0]['message']['content']
        else:
            raise Exception("Cannot extract model response text")
            
        model_text = model_text.lower()
        return model_text

    def calculate_cost(self, docs_df: pd.DataFrame):
        """
        To be used for calculating the actual cost of submitting a request to a BedRock LLM
        The provided dataframe must contain the columns 'input_tokens' and 'output_tokens'

        Args:
            docs_df: (pd.Dataframe): a dataframe containing the documents you want to calculate the cost for
         Returns:
            pd.DataFrame: the original dataframe with new columns containing input and output costs
        """

        self.logger.info("Calculating the actual cost of your dataframe")
        
        if not all(col in docs_df.columns for col in ["input_tokens","output_tokens"]):
            raise Exception("The provided dataframe does not contain the required columns for cost calculation")
            
        docs_df["actual_input_cost"] = docs_df["input_tokens"].apply(self._calc_input_cost)
        docs_df["actual_output_cost"] = docs_df["output_tokens"].apply(self._calc_output_cost)
        
        return docs_df
            
    
    def estimate_cost(self, docs_df: pd.DataFrame, text_column:str) -> tuple[pd.DataFrame, float]:
        """
        To be used for estimating the input and output cost of submitting a request to a BedRock LLM.
        Note that the input cost should be fairly accurate, whilst the output cost is an estimation 
        based on the expected json output from the model.
        
        Args:
            docs_df (pd.Dataframe): a dataframe containing the documents you want to estimate the cost for
            text_column (string): Name of the column you want to estimate the cost from.
        Returns:
            docs_df: the original dataframe with new columns containing the full model prompt, the tokenized prompt and the token length
        """
        self.logger.info("Estimating the cost of your dataframe")

        try:
            tokenizer = self._get_tokenizer()
        except KeyError:
            message = "Cost estimation not supported - model_config is missing the huggingface_path | Skipping estimation..."
            self.logger.exception(message)
            return docs_df

        docs_df["model_prompt"] = docs_df.apply(lambda row: self._create_req_body(row, text_column), axis=1)
        docs_df["tokenized_prompt"] = docs_df["model_prompt"].apply(lambda x: tokenizer(x)["input_ids"])
        docs_df["token_len"] = docs_df["tokenized_prompt"].apply(len)
        docs_df["est_input_cost"] = docs_df["token_len"].apply(self._calc_input_cost)
        docs_df["est_output_tokens"] = len(tokenizer(self.model_config["example_expected_output"])["input_ids"])
        docs_df["est_output_cost"] = docs_df["est_output_tokens"].apply(self._calc_output_cost)

        return docs_df

    def record_cost(self, result_df: pd.DataFrame):
        """
        This function logs the cost of each pipeline run to S3. It logs:
        - timestamp
        - model_id
        - model_args
        - num_records
        - total_cost

        Args:
            result_df (pd.DataFrame): dataframe containing columns actual_input_cost and actual_output_cost
        """
        cost_df = load_dataframe_from_s3(self.bucket_name, self.cost_logs)
        
        try:
            result_df = add_columns(result_df, "actual_input_cost", "actual_output_cost", "total_cost")
            total_cost = result_df["total_cost"].sum()
        except KeyError:
            total_cost = None
            
        num_records = len(result_df)
            
        new_row = pd.DataFrame({
            "timestamp": pd.Timestamp.now(),
            "model_id": self.model_id, 
            "model_args": json.dumps(self.model_args), 
            "num_records":num_records,
            "total_cost": total_cost
        }, index = [0])
        cost_df = pd.concat([new_row, cost_df], ignore_index=True)
        
        save_dataframe_to_s3(cost_df, self.bucket_name, self.cost_logs)
        self.logger.info("Pipeline cost saved in S3")

    def process_row(self, row: pd.Series) -> dict:
        """
        Processes one row of data and adds new keys/columns from the outputs of the pipeline.

        Args:
            row (pd.Series|dict): a row of data which must include a free-text column name.
        Returns:
            dict: dictionary with outputs from each pipeline stage.
        """
        text = row[self.free_text_col]
        doc_id = row.get(self.id_col)
        
        preprocessed = self.preprocessor.process(text, doc_id)

        self.logger.info(f"Document id: {doc_id} | Processing row")

        result = {
            self.id_col: doc_id,
            self.free_text_col: text,
            f"preprocessed_{self.free_text_col}": preprocessed
        }

        model_response, response_meta_data = self.model_request.call(preprocessed, doc_id)
        model_text = self._get_model_output_text(model_response)
        parsed_output = self.parse_output(model_text, doc_id)
        
        validated_output, status = self.validate_update_json(parsed_output, doc_id)
            
        result.update({
            "model_output": model_text,
            "input_tokens": int(response_meta_data["HTTPHeaders"]["x-amzn-bedrock-input-token-count"]),
            "output_tokens": int(response_meta_data["HTTPHeaders"]["x-amzn-bedrock-output-token-count"]),
            "parsed_output": parsed_output,
            "validated_output": validated_output
        })
        if status == vconf.validation_failed_message:
            #if validation failed
            result = {**result, "status":status}
        else:
            #if validation was successful
            result = {**result, **validated_output, "status":status}
        
        return result

    def run(self, df: pd.DataFrame, estimate_cost: bool = False, calculate_cost:bool = True) -> pd.DataFrame:
        """
        Takes in a whole dataframe and runs the whole pipeline on it.

        Args:
            df (pd.DataFrame): Passes a dataframe to the pipeline which has a free-text column defined.
            estimate_cost (bool): If True, estimated cost is added to the processed data frame.
            calculate_cost (bool): If True, actual cost is added to the processed data frame.
        Returns:
            processed_df (dict): Returns a new dataframe containing all values extracted from the pipeline.
        """
        self.logger.info("Running the extractor pipeline")
        
        processed = df.apply(self.process_row, axis=1)
        processed_df = pd.DataFrame(processed.tolist())
        # N.b if we want to merge processed and df together join by id.

        if estimate_cost == True:
            processed_df = self.estimate_cost(processed_df, text_column=f"preprocessed_{self.free_text_col}")
        if calculate_cost == True:
            processed_df = self.calculate_cost(processed_df)
            
        self.logger.info("Pipeline finished!")

        if self.record_pipeline_cost: self.record_cost(processed_df)
        
        return processed_df

    def parse_output(self, model_text: str, doc_id: int) -> dict:
        """
        This function attempts to parse the string output from the LLM into a JSON using langchain and regex.

        Args:
            model_text (str): the text output from the LLM
            doc_id (int): the id of the document
        Returns:
            dict: structured json output
        """
              
        try:
            # try to parse using langchain function
            model_text = model_text.strip()
            parsed = parse_json_markdown(model_text)
            self.logger.info(f"Document: {doc_id} | Output parsed from markdown format")
            if not isinstance(parsed, dict):
                raise Exception
            return parsed
        except Exception:
            # try to find json using regex
            model_text = model_text.replace("```","")
            model_text = model_text.replace("json","")
            model_text = model_text.replace("\n","")
            json_match = re.search(r'\{.*\}', model_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    self.logger.info(f"Document: {doc_id} | Output parsed using regex") 
                    return parsed
                except Exception as e:
                    error_msg = f"Document: {doc_id} | Couldn't load JSON from the regex match. Error {e}"
                    self.logger.exception(error_msg)
                    return {"error":"error when loading from regex"}
            else:
                error_msg = "Couldn't find a valid JSON in the model output."
                self.logger.exception(error_msg)
                return {"error":"no json found"}

    def validate_update_json(self, parsed_json: dict, doc_id: int) -> Tuple[dict, bool]:
        """
        This function checks a json has exactly the right keys and that the values have 
        the correct type and are allowed as per the valid_structure. Other factors such 
        as missing or extra keys and type discrepancies are handled according to the 
        BaseModel subclass provided.

        Args:
            parsed_json (dict): the json to be validated
            doc_id (int): the id of the document
        Returns:
            dict: the validated json with changes made if necessary
            bool: True or False indicating a success or failure of the validation
        """
        if "error" in list(parsed_json.keys()):
            return None, vconf.validation_failed_message
        try:
            validated  = self.valid_structure(**parsed_json)
            parsed_and_validated = validated.model_dump()

            status = check_valid_values(parsed_and_validated, self.accepted_values)

            return parsed_and_validated, status
                
        except Exception as err:
            self.logger.exception(f"Document: {doc_id} | The following error was encountered during validation:\n{err}")
            print(err)
            return parsed_json, vconf.validation_failed_message

    
            
        
        