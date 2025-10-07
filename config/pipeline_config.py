import logging

log_enable_console = False
log_enable_file = True
console_log_level = logging.INFO
log_dir = "logs"

prompt_layout = """
You have this document:
{document}

I would like you to extract out only ER Status, ER Score, PR Status, PR Score 
and HER2 Status and return the output in a JSON markdown structure. Exclude explanations and extra information from your response.

Every entity extracted must have a value from the accepted values below:

{accepted_values}

You should interpret 'weak positive', 'strong positive', 'weak negative' and so on as just 'positive' or 'negative'
Values for ER and PR score will often follow the term 'Quick Score' or 'QS'
If you can't find one of the entities then set the relevant key to null.
"""

status_values = ["positive", "negative", "not performed"]
her2_status_values = ["negative (unknown)", "negative (0)", "negative (1+)", "borderline (2+)", "positive (3+)", "not performed"]
score_values = ["0","2","3","4","5","6","7","8"]

accepted_values = {
    "er_status": status_values,
    "er_score": score_values,
    "pr_status": status_values,
    "pr_score": score_values,
    "her2_status": her2_status_values}

final_accepted_values = {
    "er_status_p": status_values,
    "er_score_p": score_values,
    "pr_status_p": status_values,
    "pr_score_p": score_values,
    "her2_status_p": her2_status_values}

examples = [{"text":"This is an example report", "entities":{
    "er_status": "positive",
    "er_score": 8,
    "pr_status": "negative",
    "pr_score": 3,
    "her2_status": "borderline"}}]

numeric_cols = ["er_score","pr_score"]
status_cols = ["er_status","pr_status","her2_status"]

cost_aggregation = 1000
mistral_7b_instruct_costs = {"input_cost_p1000":0.0002,"output_cost_p1000":0.00026}
llama3_7b_instruct_costs = {"input_cost_p1000":0.00039,"output_cost_p1000":0.00078}
claude_3_7_sonnet_costs = {"input_cost_p1000":0.003,"output_cost_p1000":0.015}
claude_3_haiku_costs = {"input_cost_p1000":0.00025,"output_cost_p1000":0.00125}

example_expected_output = """Here is your json: ```json\n{"ER Status": "positive","ER Score": 7,"PR Status": "negative","PR Score": 2,"HER2 Status": "unknown"}\n```"""

model_config_all = {
    "mistral.mistral-7b-instruct-v0:2": {"costs": mistral_7b_instruct_costs, 
                                         "example_expected_output": example_expected_output,
                                         "huggingface_path": "mistralai/Mistral-7B-Instruct-v0.2"
                                        },
    "meta.llama3-8b-instruct-v1:0": {"costs": llama3_7b_instruct_costs,
                                    "example_expected_output": example_expected_output,
                                     "huggingface_path": "meta-llama/Meta-Llama-3-8B-Instruct"
                                    },
    "anthropic.claude-3-7-sonnet-20250219-v1:0": {"costs": claude_3_7_sonnet_costs,
                                    "example_expected_output": example_expected_output,
                                    },
    "anthropic.claude-3-haiku-20240307-v1:0": {"costs": claude_3_haiku_costs,
                                    "example_expected_output": example_expected_output,
                                    }
}

placeholder_list = r'{NAME}|{INITIALS}|{NHS\s+NUMBER}|{OTHER\s+IDS}|{AGE}|{GENDER}|{RELIGION}|{ETHNICITY}|{OCCUPATION}|{EMAIL}|{PHONE\s+NUMBER}|{DATE}|{ORG\s+NAME}|{ORG\s+CODE}|{WARD\s+INFORMATION}|{ADDRESS}|{OTHER}'