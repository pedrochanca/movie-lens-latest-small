import pandas as pd
import numpy as np
from typing import Optional, Any, Dict

def load_csv_data(file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Load CSV data from a given file path.

    Parameters:
    - file_path (str): The path to the CSV file.
    - **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
    - DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def descriptive_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform descriptive analysis on a DataFrame with numerical and categorical columns.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - summary (dict): A dictionary containing summary statistics.
    """
    summary = {}

    # Numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary['numerical'] = df[numerical_cols].describe()

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = {
            'unique_values': df[col].nunique(),
            'top_value': df[col].mode()[0],
            'top_frequency': df[col].value_counts().iloc[0],
            'missing_values': df[col].isnull().sum()
        }
    
    summary['categorical'] = pd.DataFrame(categorical_summary)

    return summary
