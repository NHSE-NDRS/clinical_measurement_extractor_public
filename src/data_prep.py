import pandas as pd
import re
from config.pipeline_config import placeholder_list

def remove_gap(row: pd.Series, col:str):
    """
    Removes the term '<GAP>' from the given text.

    Args:
        row (pd.Series): the row of data to apply the function to
        col (str): the name of the column to do the replacement in
    Returns:
        pd.Series: the amended row
    """
    row[col] = row[col].replace("<GAP>","")
    return row

def remove_placeholders(row: pd.Series, col: str, by_list = True) -> pd.Series:
    """
    Remove placeholders from the given text. They can be removed by checking against a 
    deterministic list to capture specific placeholders or by a regex pattern to remove 
    any capital letters in curly brackets. Note that using the regex pattern (by_list=False)
    risks removing text that we don't want to.

    Args:
        row (pd.Series): the row of data to apply the function to
        col (str): the name of the column to do the replacement in
        by_list (bool): whether to use the default list defined in config, defaults to True

    Returns:
    pd.Series: the ammended row
    """
    if by_list:
        pattern = placeholder_list
    else:
        pattern = r'\{([A-Z\s]+)\}'
        
    row[col] = re.sub(pattern, "", row[col])
    return row