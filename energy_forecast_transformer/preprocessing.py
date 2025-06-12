import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """Load raw data and clean it (handle missing values, type conversion)."""
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    # Example: Replace missing values with NaN and convert numeric columns
    df.replace('?', np.nan, inplace=True)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df
