import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Compute mean squared error between actual and predicted values

def compute_mse(actual, predicted):
    return mean_squared_error(actual, predicted)

# Compute MSE for each variable and the mean MSE across all variables

def compute_mean_mse(multi_variate_test_dataset, forecasts, prediction_length):
    """
    Computes the MSE for each variable and the mean MSE across all variables.
    """
    mse_values = []
    for i in range(7):
        actual = multi_variate_test_dataset[0]["target"][i, -prediction_length:]
        predicted = forecasts[0, ..., i].mean(axis=0)
        mse = compute_mse(actual, predicted)
        mse_values.append(mse)
    return mse_values, np.mean(mse_values)

# Plot actual vs. predicted forecasts for each variable, including confidence intervals

def plot_forecasts(mv_indices, tags, multi_variate_test_dataset, forecasts, prediction_length, FieldName):
    """
    Plots the actual and predicted values for each variable, including mean and +/- 1 std confidence intervals.
    """
    if not isinstance(mv_indices, (list, tuple)):
        mv_indices = [mv_indices]
    n = len(mv_indices)
    fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
    if n == 1:
        axs = [axs]
    index = pd.period_range(
        start=multi_variate_test_dataset[0][FieldName.START],
        periods=len(multi_variate_test_dataset[0]["target"][0]),
        freq=multi_variate_test_dataset[0][FieldName.START].freq,
    ).to_timestamp()
    for ax, mv_index in zip(axs, mv_indices):
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.plot(
            index[-2 * prediction_length :],
            multi_variate_test_dataset[0]["target"][mv_index, -2 * prediction_length :],
            label="actual",
        )
        ax.plot(
            index[-prediction_length:],
            forecasts[0, ..., mv_index].mean(axis=0),
            label="mean",
        )
        ax.fill_between(
            index[-prediction_length:],
            forecasts[0, ..., mv_index].mean(0)
            - forecasts[0, ..., mv_index].std(axis=0),
            forecasts[0, ..., mv_index].mean(0)
            + forecasts[0, ..., mv_index].std(axis=0),
            alpha=0.2,
            interpolate=True,
            label="+/- 1-std",
        )
        ax.set_title(f"Variable {tags[mv_index]}")
        ax.legend()
    fig.autofmt_xdate()
    plt.show()

