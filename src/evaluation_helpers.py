import pandas as pd

def add_complete_match(df: pd.DataFrame, col_1: str, col_2: str, new_col: str) -> pd.DataFrame:
    """
    Adds a boolean column on whether the row values match across the two columns.
    If col_1 has value A and col_2 has value A this will return True. 

    Args:
        df (pd.DataFrame): DataFrame containing columns to compare.
        col_1 (str): First column name.
        col_2 (str): Second column name.
        new_col (str): Name for the new boolean column.

    Returns:
        pd.DataFrame: DataFrame with the new boolean column added.
    """

    if df[col_1].dtype != df[col_2].dtype:
        raise TypeError("Data types in the two columns do not match.")

    df[new_col] = df[col_1].eq(df[col_2]) | (df[col_1].isna() & df[col_2].isna())

    return df

def calculate_column_difference(df: pd.DataFrame, column1: str, column2: str, column_name: str) -> pd.DataFrame:
    """
    Calculate the difference between two columns in a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data
        column1 (str): Name of the first column (minuend)
        column2 (str): Name of the second column (subtrahend)
        column_name (str): Name for the new difference column to be created
        
    Returns:
        pd.DataFrame: DataFrame with the new difference column added
    """
    df[column_name] = df[column1] - df[column2]
    return df



def add_columns(df: pd.DataFrame, column_1: str, column_2: str, column_name: str) -> pd.DataFrame:
    """
    Adds the value of two columns in a pandas DataFrame and creates a new column with the result.    
    
   Args:
       df (pd.DataFrame):  The DataFrame
       column_1 (str): Name of the first column
       column_2 (str): Name of the second column
       column_name (str): Name of the new column to store the sum
        
    Returns:
        pd.DataFrame: DataFrame with the new column added
    """
    df[column_name] = df[column_1] + df[column_2]
    return df


