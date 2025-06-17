# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load and clean the raw dataset, handling missing values and type conversion

def load_and_clean_data(filepath):
    """
    Load raw data and clean it (handle missing values, type conversion).
    Returns a DataFrame indexed by datetime.
    """
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    df.replace('?', np.nan, inplace=True)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'Time'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.set_index('datetime')
    return df

# Fill missing values in the first 7 columns using linear interpolation

def fill_missing_values(df):
    """
    Fill missing values in the first 7 columns using linear interpolation.
    """
    df.iloc[:, :7] = df.iloc[:, :7].interpolate(method='linear')
    return df

# Resample the dataframe to a different frequency (e.g., daily)

def resample_data(df, freq='D'):
    """
    Resample the dataframe to the given frequency (default: daily).
    """
    return df.resample(freq).mean()

# Normalize selected columns of the dataframe using MinMaxScaler

def normalize_data(df, columns=None):
    """
    Normalize selected columns of the dataframe using MinMaxScaler.
    Returns the normalized DataFrame and the scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    if columns is None:
        columns = df.columns.tolist()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# Create dictionaries for train, validation, and test splits for use with HuggingFace Datasets

def create_dataset_dict(df_daily, n_train, n_val, n_test, prediction_length):
    """
    Organize the data into dictionaries for training, validation, and testing.
    Each dictionary contains time series data, static features, and item identifiers.
    """
    import pandas as pd
    data_train = {
        'start': [pd.Timestamp('2006-12-16 00:00:01')] * 7,
        'target': [
            df_daily['Global_active_power'].iloc[:n_train].values.tolist(),
            df_daily['Global_reactive_power'].iloc[:n_train].values.tolist(),
            df_daily['Voltage'].iloc[:n_train].values.tolist(),
            df_daily['Global_intensity'].iloc[:n_train].values.tolist(),
            df_daily['Sub_metering_1'].iloc[:n_train].values.tolist(),
            df_daily['Sub_metering_2'].iloc[:n_train].values.tolist(),
            df_daily['Sub_metering_3'].iloc[:n_train].values.tolist()
        ],
        'feat_static_cat': [[0], [1], [2], [3], [4], [5], [6]],
        'feat_dynamic_real': [None] * 7,
        'item_id': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    }
    data_val = {
        'start': [pd.Timestamp('2006-12-16 00:00:01')] * 7,
        'target': [
            df_daily['Global_active_power'].iloc[:n_val].values.tolist(),
            df_daily['Global_reactive_power'].iloc[:n_val].values.tolist(),
            df_daily['Voltage'].iloc[:n_val].values.tolist(),
            df_daily['Global_intensity'].iloc[:n_val].values.tolist(),
            df_daily['Sub_metering_1'].iloc[:n_val].values.tolist(),
            df_daily['Sub_metering_2'].iloc[:n_val].values.tolist(),
            df_daily['Sub_metering_3'].iloc[:n_val].values.tolist()
        ],
        'feat_static_cat': [[0], [1], [2], [3], [4], [5], [6]],
        'feat_dynamic_real': [None] * 7,
        'item_id': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    }
    data_test = {
        'start': [pd.Timestamp('2006-12-16 00:00:01')] * 7,
        'target': [
            df_daily['Global_active_power'].iloc[:].values.tolist(),
            df_daily['Global_reactive_power'].iloc[:].values.tolist(),
            df_daily['Voltage'].iloc[:].values.tolist(),
            df_daily['Global_intensity'].iloc[:].values.tolist(),
            df_daily['Sub_metering_1'].iloc[:].values.tolist(),
            df_daily['Sub_metering_2'].iloc[:].values.tolist(),
            df_daily['Sub_metering_3'].iloc[:].values.tolist()
        ],
        'feat_static_cat': [[0], [1], [2], [3], [4], [5], [6]],
        'feat_dynamic_real': [None] * 7,
        'item_id': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    }
    return data_train, data_val, data_test

from functools import lru_cache

# Convert a date string to a pandas Period object for time series indexing

def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

# Transform the 'start' field in a batch to pandas Periods

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch