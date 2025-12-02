import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch

# Import TiRex
from tirex import ForecastModel, load_model

# Settings
DATASET_DIR = "dataset"
SYMBOL = "AAPL"  # You can change this to any symbol you have data for
CONTEXT_LENGTH = 60  # Number of past days to use for prediction
FORECAST_LENGTH = 1  # Predict 1 day ahead

# Load test data
def load_test_data(symbol, context_length, forecast_length):
    path = os.path.join(DATASET_DIR, f"{symbol}_25y_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df = df.sort_values("Date").reset_index(drop=True)
    close_prices = df["Close"].values.astype(np.float32)
    # Split: last 15 years for test (approx 252*15 trading days)
    test_days = 252 * 15
    test_data = close_prices[-test_days - context_length:]
    X_test, y_test = [], []
    for i in range(context_length, len(test_data) - forecast_length + 1):
        X_test.append(test_data[i-context_length:i])
        y_test.append(test_data[i:i+forecast_length])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # Reshape for univariate: [batch, context_length, 1]
    X_test = X_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    return X_test, y_test, df

# Load model
def load_tirex_model():
    # This will download and load the pretrained model from Hugging Face
    model = load_model("NX-AI/TiRex")
    model.eval()
    return model

# Inference
def run_inference(model, X_test, forecast_length):
    preds = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            ctx = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            quantiles, mean = model.forecast(ctx, prediction_length=forecast_length)
            # mean shape: [batch, prediction_length, 1] or [batch, prediction_length]
            # We use mean[0, 0] for 1-step ahead
            preds.append(mean[0, 0] if forecast_length == 1 else mean[0])
    preds = np.array(preds)
    if forecast_length == 1:
        preds = preds.reshape(-1, 1, 1)
    return preds

# Evaluation
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    direction_acc = np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten())))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Directional Accuracy: {direction_acc:.4f}")
    return rmse, mae, r2, direction_acc

# Backtesting (simple strategy: buy if next day predicted > today)
def backtest(y_true, y_pred):
    signals = (y_pred.flatten() > np.roll(y_true.flatten(), 1)).astype(int)
    returns = np.diff(y_true.flatten()) / y_true.flatten()[:-1]
    strategy_returns = returns * signals[1:]
    cumulative = np.cumprod(1 + strategy_returns)
    plt.figure(figsize=(12,6))
    plt.plot(np.cumprod(1 + returns), label="Buy & Hold")
    plt.plot(cumulative, label="TiRex Strategy")
    plt.title("Backtest: Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Strategy Total Return: {cumulative[-1]:.2f}x")
    print(f"Buy & Hold Return: {np.cumprod(1 + returns)[-1]:.2f}x")

if __name__ == "__main__":
    print("Loading test data...")
    X_test, y_test, df = load_test_data(SYMBOL, CONTEXT_LENGTH, FORECAST_LENGTH)
    print(f"Test samples: {X_test.shape[0]}")
    print("Loading TiRex model...")
    model = load_tirex_model()
    print("Running inference...")
    preds = run_inference(model, X_test, FORECAST_LENGTH)
    print("Evaluating...")
    evaluate(y_test, preds)
    print("Backtesting...")
    backtest(y_test, preds) 