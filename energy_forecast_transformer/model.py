from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_time_series_windows(data, window_size, target_col):
    """Erstellt Input- und Output-Fenster für Zeitreihenprognose."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

def build_transformer_model(input_shape):
    """Erstellt ein einfaches Transformer-Modell für Zeitreihen."""
    inputs = keras.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model
