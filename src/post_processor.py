import pandas as pd
import re
import numpy as np

import config.pipeline_config as conf
import config.validation_config as vconf
from src.helpers import check_valid_values

class PostProcessor:
    """
    Class containing the post-processing functions

    Args:
        df: the dataframe to post-process
        cols_to_process: list of columns that you want post-processing to apply to
    """
    def __init__(self,
                 df: pd.DataFrame,
                 cols_to_process: list):

        self.df = df
        self.cols_to_process = cols_to_process
        
    def apply_general_mapping(self, row: dict, mapping: dict, cols_to_map: list) -> dict:
        """
        Maps new values to original values based on adjustable dictionary
        
        If no match is found, the original value is kept.

        Args:
            row (dict): The input dictionary representing one row of data.
            mapping (dict): A dictionary mapping raw values to new values.
            cols_to_map (list): List of column names to apply the mapping to.
        Returns:
            dict: a dictionary with the values updated
        """
        updated_row = row.copy()
        for col in cols_to_map:
            val = row.get(col)
            if val in [np.nan, None]: continue
            val_norm = val.strip().lower()
            updated_row[col] = mapping.get(val_norm, val)
        return updated_row
    
    def map_score(self, row: dict, cols_to_map: list) -> dict:
        """
        converts score values in the format 'x/8' (where x is 0–8) to just 'x'

        Args:
            row (dict): A dictionary representing a single data row with string values
            cols_to_map (list): A list of keys (columns) to check and potentially convert
        Returns:
            dict: a dictionary with the values updated
        """
        updated_row = row.copy()
        for col in cols_to_map:
            val = row.get(col)
            if val in [np.nan, None]: continue
            val_norm = val.strip().lower()
            if re.fullmatch(r"[0-8]/8", val_norm):
                updated_row[col] = val_norm.split("/")[0]
        return updated_row

    def map_two_part_scores(self, original: dict, cols_to_check: list) -> dict:
        """
        Converts a dictionary with values like '5+3', '2+2' to their single-digit equivalent.
        Only keys specified in cols_to_check will be converted.

        Args:
            original (dict): a dictionary containing the original values
            cols_to_check (list): a list of keys (columns) to do the conversion on
        Returns:
            dict: a dictionary with the values updated
        """
        
        new_values = original.copy()
        pattern = re.compile(r'^\s*\d\s*\+\s*\d\s*$')
        for col in cols_to_check:
            if original[col] in [np.nan, None]: continue
            if pattern.match(original[col]):
                number = eval(original[col])
                new_values[col] = str(number)
        return new_values
        
    def score_to_status(self, row: dict, pairs: list[tuple[str,str]]) -> dict:
        """ 
        Takes a row and convert the ER and PR status columns
        based on the score.
        Args:
           row (dict): row of data
           pairs (list): list of tuples with pairs of columns
        Returns:
           dict: a new dictionary with mapped values
        """
        #make a copy so the input dict is not mutated
        result = dict(row)
        # map the scores to expected status
        mapping = {"0": "negative", "8": "positive"}
        
        #iterate over the column name pairs,/act if both exists 
        for score_col,status_col in pairs:
             #act if both keys exists 
            if score_col in row and status_col in row:
                val =row.get(score_col)
                if pd.isna(val):
                    continue
                score_text = val.strip()
                if score_text in mapping:
                    result[status_col] = mapping[score_text]
    
        return result

    def process_row(self, original_row: dict | pd.Series) -> dict:
        """
        Processes one row of data and adds new columns with post-processed values.
        New columns are suffixed with '_p'

        Args:
            row (dict or pd.Series): row of data
        Returns:
            dict: Returns a new dictionary with new columns appended.
        """
        data_to_process = original_row[self.cols_to_process]

        # no need to map anything if original validation failed
        if original_row['status'] == vconf.validation_failed_message:
            return {**original_row, "status_processed": vconf.validation_failed_message}

        new_dict = self.map_two_part_scores(data_to_process, conf.numeric_cols)
        new_dict = self.score_to_status(new_dict, [("er_score","er_status"),("pr_score","pr_status")])

        new_dict = self.map_score(new_dict, conf.numeric_cols)
        
        her2_mapping = {"0": "negative", "1+": "negative", "2+": "borderline", "3+": "positive"}
        new_dict = self.apply_general_mapping(new_dict, her2_mapping, ["her2_status"])

        common_errors = {"null": np.nan}
        new_dict = self.apply_general_mapping(new_dict, common_errors, conf.numeric_cols + conf.status_cols)            

        new_dict = {k: None if isinstance(v, float) and np.isnan(v) else v for k,v in new_dict.items()}#convert nan to None ready for re-validation
        
        status_processed = check_valid_values(new_dict, conf.accepted_values)
        
        new_dict = {f"{k}_p": v for k, v in new_dict.items()}

        return {**original_row, **new_dict, "status_processed":status_processed}

    def run(self):
        """
        Runs all postprocessing steps on the dataframe and columns provided in the class.

        Returns:
            pd.DataFrame: df with new post-processed columns
        """
        processed = self.df.apply(self.process_row, axis = 1)
        processed_df = pd.DataFrame(processed.tolist())

        return processed_df
    
