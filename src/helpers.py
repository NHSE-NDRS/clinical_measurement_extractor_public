import yaml
import logging
import os
from huggingface_hub import login
from dotenv import load_dotenv

import config.pipeline_config as conf
import config.validation_config as vconf

def load_config_from_yaml(file_path: str) -> dict:
    """
    Load configuration settings from a YAML file.

    Args:
        file_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML config file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def hf_login(logger: logging.Logger):
    """
    Login to HuggingFace to load LLM Models/Tokenizers

    Args:
        logger (logging.Logger): This is an initialised logger for logging information to the user.

    """
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    try:
        login(hf_token)
        logger.info("HuggingFace authentication successful")
    except Exception as e:
        logger.exception(f"There was a problem with authentication to HuggingFace\nError: {e}")
        raise

def check_valid_values(input_json: dict, accepted_values: dict) -> str:
    """
    Checks whether the values in the input_json are valid according to
    pipeline config and validation config.

    Args:
        input_json (dict): the dictionary to check
        accepted_values (dict): contains a list of the accepted values for each key in the input json

    Returns:
        str: a status of 'valid', 'partial' or 'invalid'
    """
    
    valid_indicators = [input_json[k] in accepted_values[k] + [None] for k in accepted_values.keys()]
    
    if all(i == True for i in valid_indicators):
        status = vconf.valid_status
    elif all(i == False for i in valid_indicators):
        status = vconf.none_valid_status
    else:
        status = vconf.partial_valid_status

    return status