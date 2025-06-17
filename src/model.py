import numpy as np
import torch
from transformers import InformerConfig
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error



def get_informer_config(num_of_variates, prediction_length, context_length, lags_sequence, time_features):
    """
    Returns an InformerConfig object with the specified parameters.
    Args:
        num_of_variates: Number of input features (time series).
        prediction_length: Number of time steps to predict.
        context_length: Number of past time steps used for prediction.
        lags_sequence: List of lag indices for lagged features.
        time_features: List of time feature names (e.g., day_of_week).
    """
    return InformerConfig(
        input_size=num_of_variates,
        prediction_length=prediction_length,
        context_length=context_length,
        lags_sequence=lags_sequence,
        num_time_features=len(time_features) + 1,
        dropout=0.05,
        encoder_layers=4,
        decoder_layers=2,
        d_model=128,
    )

# Main training loop for the Informer model
# Handles training, validation, early stopping, and loss tracking

def train_model(model, train_dataloader, test_dataloader, multi_variate_test_dataset, config, prediction_length, epochs=20, patience=5, num_batches_per_epoch=100):
    """
    Trains the Informer model using the provided dataloaders and configuration.
    Args:
        model: The InformerForPrediction model instance.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for validation/test data.
        multi_variate_test_dataset: List of test dataset dictionaries.
        config: InformerConfig object.
        prediction_length: Number of time steps to predict.
        epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before early stopping.
        num_batches_per_epoch: Number of batches per epoch.
    Returns:
        loss_history: List of training loss values per batch.
        val_loss_history: List of validation MSE values per epoch.
    """

    loss_history = []  # Stores training loss for each batch
    val_loss_history = []  # Stores validation MSE for each epoch
    best_val_loss = float('inf')  # Track the best validation loss for early stopping
    counter = 0  # Early stopping counter
    train_loss_history = []
    
    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Prepare model, optimizer, and dataloader for distributed/accelerated training
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        batch_count = 0
        for _, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
                static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            # Backward pass and optimization
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            loss_history.append(loss.item())

        # Normalize the training loss by the actual number of batches
        train_loss /= batch_count

        # Validation: generate predictions and compute MSE
        model.eval()
        val_mse = 0
        forecasts = []
        actual_values = multi_variate_test_dataset[0]["target"]  # Ground truth values
        with torch.no_grad():
            for batch in test_dataloader:
                outputs = model.generate(
                    static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
                    static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
                    past_time_features=batch["past_time_features"].to(device),
                    past_values=batch["past_values"].to(device),
                    future_time_features=batch["future_time_features"].to(device),
                    past_observed_mask=batch["past_observed_mask"].to(device),
                )
                forecasts.append(outputs.sequences.cpu().numpy())

        forecasts = np.vstack(forecasts)
        for i in range(7):  # Loop over all features/variables
            predicted = forecasts[0, ..., i].mean(axis=0)
            actual = actual_values[i, -prediction_length:]  # Last prediction_length values as ground truth
            val_mse += mean_squared_error(actual, predicted)
        val_mse /= 7  # Average MSE across all features

        # Early stopping and learning rate scheduling
        scheduler.step(val_mse)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop = epoch
                break
        train_loss_history.append(train_loss)
        val_loss_history.append(val_mse)
    
    # workaround for printing loss history because of double printing in accelerate !??
    for i in range(stop):
        accelerator.print(f"Epoch {i+1}, Train Loss: {train_loss_history[i]:.4f}, Val MSE: {val_loss_history[i]:.4f}")    
    accelerator.print(f"Early stopping after {stop+1} epochs")
    
    return loss_history, val_loss_history


