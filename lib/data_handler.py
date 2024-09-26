import pandas as pd

def load_csv_data(file_path, **kwargs):
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