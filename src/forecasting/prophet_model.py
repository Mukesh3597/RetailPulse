from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_and_forecast(horizon=30):
    # 1. Check input file
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load and prepare data
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    df = df[["Date", "Total_Revenue"]].copy()
    df = df.rename(columns={"Date": "ds", "Total_Revenue": "y"})

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Combine duplicate dates, if any
    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) <= horizon:
        raise ValueError(
            f"Need more than {horizon} days of data. " f"Available rows: {len(df)}"
        )

    print("Total historical days:", len(df))
    print("First date:", df["ds"].min().date())
    print("Last historical date:", df["ds"].max().date())

    # 3. Hold out the last 30 days for evaluation
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    print("Training rows:", len(train))
    print("Test rows:", len(test))

    # 4. Train Prophet
    model = Prophet(
        weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False
    )

    model.fit(train)

    # 5. Evaluate on the held-out 30 days
    # Use historical dates plus test dates for evaluation.
    evaluation_dates = pd.DataFrame({"ds": test["ds"].to_numpy()})

    evaluation_forecast = model.predict(evaluation_dates)

    predictions = test.merge(evaluation_forecast[["ds", "yhat"]], on="ds", how="left")

    predictions["yhat"] = predictions["yhat"].clip(lower=0)

    mae = mean_absolute_error(predictions["y"], predictions["yhat"])

    rmse = np.sqrt(mean_squared_error(predictions["y"], predictions["yhat"]))

    nonzero = predictions["y"] != 0

    if nonzero.any():
        mape = (
            np.abs(
                (predictions.loc[nonzero, "y"] - predictions.loc[nonzero, "yhat"])
                / predictions.loc[nonzero, "y"]
            ).mean()
            * 100
        )
    else:
        mape = np.nan

    predictions.to_csv(REPORTS_DIR / "prophet_forecast_results.csv", index=False)

    metrics = pd.DataFrame(
        [{"Model": "Prophet", "MAE": mae, "RMSE": rmse, "MAPE_percent": mape}]
    )

    metrics.to_csv(REPORTS_DIR / "prophet_metrics.csv", index=False)

    # 6. Train a final model on all available historical data
    final_model = Prophet(
        weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False
    )

    final_model.fit(df)

    # 7. Generate the next 30 calendar days
    future_dates = pd.date_range(
        start=df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    future = pd.DataFrame({"ds": future_dates})
    future_forecast = final_model.predict(future)

    future_forecast = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Revenue cannot be negative
    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        future_forecast[column] = future_forecast[column].clip(lower=0)

    # 8. Save the future forecast
    future_path = REPORTS_DIR / "prophet_future_forecast.csv"

    future_forecast.to_csv(future_path, index=False)

    # 9. Display results
    print("\n--- Prophet Evaluation Results ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (% on non-zero actual days): {mape:.2f}")

    print("\n--- Future Forecast ---")
    print("Future forecast rows:", len(future_forecast))
    print("Forecast starts:", future_forecast["ds"].min().date())
    print("Forecast ends:", future_forecast["ds"].max().date())
    print("Future forecast saved to:", future_path)

    print("\nSample future predictions:")
    print(future_forecast.head())

    return final_model, future_forecast, metrics


if __name__ == "__main__":
    train_and_forecast(horizon=30)
