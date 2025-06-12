import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Prediction')
    plt.legend()
    plt.title('Energy Consumption: Prediction vs. Actual')
    plt.show()

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
