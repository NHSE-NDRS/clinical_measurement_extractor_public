import boto3
import pandas as pd
from io import StringIO
from typing import List
from datetime import datetime

def load_dataframe_from_s3(bucket_name: str, object_key: str, str_columns: list = None) -> pd.DataFrame:
    """
    Load a CSV file from an S3 bucket into a pandas DataFrame.

    Args:
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key (path/filename) of the object in the S3 bucket.
        str_columns (list): An optional list of columns that you want to load as string types
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    s3 = boto3.client('s3')
    if str_columns:
        dtype_mapping = {item: str for item in str_columns}
    else:
        dtype_mapping = None
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), dtype = dtype_mapping)
        return df
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

def save_dataframe_to_s3(df: pd.DataFrame, bucket_name: str, object_key: str, description: str = "") -> None:
    """
    Save a pandas DataFrame as a CSV file to an S3 bucket with optional metadata.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key (path/filename) to save the object in the S3 bucket.
        description (str): Description to include in the S3 object metadata.
    """
    s3 = boto3.client('s3')

    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=csv_buffer.getvalue(),
            Metadata={'description': description}
        )
        print(f"Successfully saved DataFrame to s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"Error saving data to S3: {e}")
        raise

def save_eval_df_to_s3(df: pd.DataFrame, bucket_name: str, folder:str, description: str = ""):
    """
    Save summary_df to S3 with embedded comparison_dict.

    The object will be saved as 'evaluation_df_{timestamp}.csv' where timestamp is in
    the format YYYYMMDD_HHMMSS.

    Args:
        df (pd.DataFrame): Evaluation dataframe to save
        bucket_name (str): The name of the S3 bucket.
        folder (str): sub folder within the S3 bucket
        description (str): Optional description saved in the S3 object's metadata.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    object_key = f"{folder}/evaluation_df_{timestamp}.csv"

    save_dataframe_to_s3(df, bucket_name, object_key, description=description)

def combine_dataframe_from_list(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine a list of pandas DataFrames into a single DataFrame.

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames to combine.

    Returns:
        pd.DataFrame: A single combined DataFrame.
    """
    if not df_list:
        raise ValueError("The list of dataframes is empty.")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df