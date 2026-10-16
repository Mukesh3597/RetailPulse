
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

LOOKBACK = 30
EPOCHS = 30


class SalesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def train_and_forecast(horizon=30):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sales data not found: {DATA_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df[["Date", "Total_Revenue"]].dropna().sort_values("Date")
    values = df["Total_Revenue"].to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(values) <= LOOKBACK + horizon:
        raise ValueError("Not enough data for the selected lookback and horizon.")

    train_values = values[:-horizon]
    scaler = MinMaxScaler()
    scaler.fit(train_values)

    scaled_all = scaler.transform(values)
    train_scaled = scaled_all[:-horizon]

    X, y = [], []
    for i in range(LOOKBACK, len(train_scaled)):
        X.append(train_scaled[i - LOOKBACK:i, 0])
        y.append(train_scaled[i, 0])

    X_train = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    model = SalesLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    # Forecast the held-out 30-day period recursively
    model.eval()
    window = scaled_all[len(train_values) - LOOKBACK:len(train_values), 0].copy()
    forecast_scaled = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window, dtype=torch.float32).reshape(1, LOOKBACK, 1)
            next_value = model(x).item()
            forecast_scaled.append(next_value)
            window = np.append(window[1:], next_value)

    forecast = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten()
    forecast = np.maximum(forecast, 0)

    actual = values[-horizon:, 0]
    dates = df["Date"].iloc[-horizon:].to_numpy()

    results = pd.DataFrame({
        "Date": dates,
        "Actual_Revenue": actual,
        "LSTM_Forecast": forecast
    })
    results.to_csv(REPORTS_DIR / "lstm_forecast_results.csv", index=False)

    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )

    metrics = pd.DataFrame([{
        "Model": "LSTM",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape
    }])
    metrics.to_csv(REPORTS_DIR / "lstm_metrics.csv", index=False)

    torch.save(model.state_dict(), PROJECT_ROOT / "models" / "lstm_model.pth")

    print(f"Training sequences: {len(X_train)}")
    print(f"Test days: {len(actual)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (% on non-zero actual days): {mape:.2f}")
    print(f"Results saved: {REPORTS_DIR / 'lstm_forecast_results.csv'}")

    return model, results, metrics


if __name__ == "__main__":
    train_and_forecast(horizon=30)
