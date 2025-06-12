import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """Lädt die Rohdaten und bereinigt sie (fehlende Werte, Typumwandlung)."""
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    # Beispiel: Ersetze fehlende Werte durch NaN und konvertiere numerische Spalten
    df.replace('?', np.nan, inplace=True)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df
