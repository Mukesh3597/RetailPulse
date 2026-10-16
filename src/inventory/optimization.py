from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales.csv"
FORECAST_PATH = PROJECT_ROOT / "reports" / "prophet_future_forecast.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def create_inventory_recommendations(lead_time_days=7, safety_stock_days=3):
    """Create illustrative inventory recommendations from sales forecasts."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sales data not found: {DATA_PATH}")

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast not found: {FORECAST_PATH}. Run prophet_model.py first."
        )

    sales = pd.read_csv(DATA_PATH)
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["ds"])

    # Use positive daily quantities to estimate typical units sold.
    # Actual inventory on hand is not available in the dataset.
    daily_demand = sales["Total_Quantity"].fillna(0).clip(lower=0)

    avg_daily_demand = daily_demand.mean()

    # Revenue forecast is not unit demand; use historical quantity
    # as a simple proxy until a product-level forecast is available.
    safety_stock = avg_daily_demand * safety_stock_days
    reorder_point = avg_daily_demand * lead_time_days + safety_stock

    recommendations = pd.DataFrame(
        {
            "Forecast_Date": forecast["ds"],
            "Forecast_Revenue": forecast["yhat"].clip(lower=0),
            "Estimated_Avg_Daily_Units": avg_daily_demand,
            "Assumed_Lead_Time_Days": lead_time_days,
            "Safety_Stock_Units": round(safety_stock, 2),
            "Reorder_Point_Units": round(reorder_point, 2),
            "Recommendation": "Review product-level stock before reordering",
        }
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "inventory_recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    print("Inventory recommendations created:", len(recommendations))
    print("Average daily units (historical proxy):", round(avg_daily_demand, 2))
    print("Assumed lead time:", lead_time_days, "days")
    print("Safety stock (estimated units):", round(safety_stock, 2))
    print("Reorder point (estimated units):", round(reorder_point, 2))
    print("Saved to:", output_path)
    print("\nSample recommendations:")
    print(recommendations.head())

    return recommendations


if __name__ == "__main__":
    create_inventory_recommendations()
