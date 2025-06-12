import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='Echt')
    plt.plot(y_pred, label='Vorhersage')
    plt.legend()
    plt.title('Energieverbrauch: Vorhersage vs. Echt')
    plt.show()

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
