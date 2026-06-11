import pandas as pd
from src.utils.helpers import setup_logger

logger = setup_logger("data_loader")

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads the raw dataset from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    logger.info(f"Loading raw dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise e
