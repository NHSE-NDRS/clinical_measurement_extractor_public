# Clinical Measurement Extractor

This is a repository to create a LLM pipeline to extract out clinical entities from free-text data.

The pipeline is currently set up to extract five metrics from breast cancer pathology reports:
- ER Status
- ER Score
- PR Status
- PR Score
- HER2 Status

The exact metrics being extracted can be altered by changing the prompt and various config files.

## Project structure

```text
+---clinical_measurement_extractor
|   |
|   +---src                                       <- Folder that contains the code required.
|   |   +---load_data.py                          <- Functions defined to load data in from an S3 bucket.
|   |   +---helpers.py                            <- Functions defined to support loading config from a YAML file and login to HF.
|   |   +---extractor_pipeline.py                 <- Class for running the extractor pipeline.
|   |   +---model_request.py                      <- Class for requesting a response from a BedRock Hosted Model.
|   |   +---prompt_builder.py                     <- Class for building the Prompt.
|   |   +---text_preprocessor.py                  <- Class for processing the text.
|   |   +---custom_logging.py                     <- Setup script for logging
|   |   +---evaluation_helpers.py                 <- Functions to assist with evaluation
|   |   +---post_processor.py                     <- Class for post processing the outputs from the CME pipeline.
|   |   +---cme_evaluator.py                      <- Class for evaluating the outputs from the CME pipeline.
|   |   +---data_prep.py                          <- Functions for data preparation.
|   | 
|   +---tests                                     <- Folder containing the tests for the repository.
|   |   +---test_helpers.py                       <- Test the Helper Functions.
|   |   +---test_extractor_pipeline.py            <- Test the ExtractorPipeline Class.
|   |   +---test_prompt_builder.py                <- Test the PromptBuilder Class.
|   |   +---test_text_preprocessor.py             <- Test the TextPreprocessor Class.
|   |   +---test_evaluation_helpers.py            <- Test the evaluation helpers.
|   |   +---test_model_request.py                 <- Test the ModelRequest Class.
|   |   +---test_logging.py                       <- Test logging functionality.
|   |   +---test_post_processor.py                <- Test PostProcessor functions.
|   |   +---test_cme_evaluator.py                 <- Test CMEEvaluator class.
|   |   +---test_data_prep.py                     <- Test data preparation functions.
|   | 
|   +---config                                    <- Folder containing local config for the individual.
|   |   +---local.yaml                            <- YAML File with config needed per the individual.
|   |   +---pipeline_config.py                    <- Config for the running of the pipeline.
|   |   +---validation_config.py                  <- Config for the validating LLM outputs.
|   | 
|   +---logs                                      <- Folder containing the logs on each run of the pipeline.
|   | 
|   +---save_data                                 <- Folder containing notebooks to clean and save data to S3.
|   | 
|   +---eda                                       <- Folder containing notebooks with exploratory data analysis.
|   | 
|   +---clinical_measurement_extractor.ipynb      <- Jupyter notebook to run the code
|   | 
|   +---prompt_engineering.ipynb                  <- Jupyter notebook with prompt engineering workflow
|   | 
|   +---.env                                      <- env file to store tokens and IDs
|
|   README.md                                     <- Quick start guide
```

## Getting Started

### SageMaker Notebook Set-up

This code is run using JupyterLab in the SageMaker Studio.

Notebook configuration:
* **Instance**: ml.t3.medium
* **Image**: Sagemaker Distribution 3.6.1 (Supports Python 3.12.9)
* **Storage(GB)** 5

### Requesting Access to HuggingFace Repositories (optional).
This project uses OpenAI, Claude, Llama and Mistral Models.

This step is only required if you want to estimate the cost of running the pipeline using the `estimate_cost` functionality.

The code will work without setting this up.

If you want to set this up, create an account with HuggingFace, and then request access to these models:

1. [Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

### Creating a config/local.yaml file

Create a YAML file located in the config folder called "local.yaml" and give config values for:

```yaml
FREETEXT_COL: "{FREE TEXT COLUMN NAME}"
BUCKET_NAME: "{S3 BUCKET NAME}"
BATCH1_ITEM: "{ITEM NAME OF BATCH 1 DATA}"
ID_COL: "{ID COLUMN NAME}"
THE_DATA: "{DATA FOR PROMPT ENGINEERING}"
COST_LOGS: "{FILE NAME FOR COST LOGGING}"
MODEL_ID: "{MODEL ID}"
MODEL_ARGS: 
    ...
YOUR_S3_FOLDER: "{S3 FOLDER NAME}"
PROMPT_MANAGEMENT_ID: "{CME PROMPT ID}"
PROMPT_MANAGEMENT_NAME: "{PROMPT MANAGEMENT NAME}"
```

### Creating a .env file

Create a .env file. This is a hidden file, so you will need to make sure hidden files are visible.

```.env
HUGGINGFACE_TOKEN={HUGGINGFACE TOKEN}
AWS_REGION={AWS REGION}
CME_PROMPT_ID={CME PROMPT ID}
```

### Running Pytests

To run pytests in the terminal. Open a new terminal tab in the SageMaker Notebook, then cd into the clinical_measurement_extractor folder.

Then in the terminal run:
```bash
python -m pytest
```
To run one specific test file, one specific test class or one specific test function, use the following commands respectively:
```bash
python -m pytest tests/your_test_file.py
python -m pytest tests/your_test_file.py::YourTestClass
python -m pytest tests/your_test_file.py::YourTestClass::your_test_func
```
