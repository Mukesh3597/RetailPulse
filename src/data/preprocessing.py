from pathlib import Path
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import load_retail_data


def clean_retail_data(df):
    """
    Remove duplicate rows, isolate negative-price records,
    fill missing descriptions where possible, and calculate revenue.
    """
    df_clean = df.drop_duplicates().copy()

    # Keep negative-price records separate for auditing.
    bad_debt_mask = df_clean["Price"] < 0
    bad_debt_records = df_clean.loc[bad_debt_mask].copy()

    df_clean = df_clean.loc[~bad_debt_mask].copy()

    # Fill missing descriptions using the most common description
    # associated with the same StockCode.
    description_lookup = (
        df_clean.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            lambda values: (values.mode().iloc[0] if not values.mode().empty else None)
        )
    )

    missing_description = df_clean["Description"].isna()

    df_clean.loc[missing_description, "Description"] = df_clean.loc[
        missing_description, "StockCode"
    ].map(description_lookup)

    # Keep missing Customer IDs; these rows can still help
    # with overall sales forecasting.
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["Price"]

    return df_clean, bad_debt_records


def create_daily_sales(df_clean):
    """
    Aggregate positive-quantity, positive-price transactions
    into a continuous daily sales time series.
    """
    sales_df = df_clean.loc[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)].copy()

    sales_df["Date"] = sales_df["InvoiceDate"].dt.normalize()

    daily_sales = (
        sales_df.groupby("Date")
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Total_Revenue=("Revenue", "sum"),
            Total_Invoices=("Invoice", "nunique"),
        )
        .sort_index()
    )

    # Create a continuous calendar.
    # Dates without recorded transactions are provisionally set to zero.
    all_dates = pd.date_range(
        start=daily_sales.index.min(),
        end=daily_sales.index.max(),
        freq="D",
    )

    daily_sales = daily_sales.reindex(all_dates)

    sales_columns = [
        "Total_Quantity",
        "Total_Revenue",
        "Total_Invoices",
    ]

    daily_sales[sales_columns] = daily_sales[sales_columns].fillna(0)

    daily_sales.index.name = "Date"
    daily_sales = daily_sales.reset_index()

    # Flag the investigated revenue spike; do not delete it.
    daily_sales["Revenue_Spike_Flag"] = daily_sales["Date"] == pd.Timestamp(
        "2011-12-09"
    )

    return daily_sales


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "raw" / "online_retail_II.xlsx"

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load original dataset.
    raw_df = load_retail_data(input_path)

    # 2. Clean the data.
    clean_df, bad_debt_records = clean_retail_data(raw_df)

    # 3. Create daily sales dataset.
    daily_sales = create_daily_sales(clean_df)

    # 4. Save processed data.
    output_path = output_dir / "daily_sales.csv"
    daily_sales.to_csv(output_path, index=False)

    # 5. Print verification results.
    print("Preprocessing completed successfully!")
    print("Raw shape:", raw_df.shape)
    print("Clean shape:", clean_df.shape)
    print("Bad-debt records isolated:", len(bad_debt_records))
    print("Daily dataset shape:", daily_sales.shape)
    print("Daily dataset columns:", daily_sales.columns.tolist())
    print("Missing values:\n", daily_sales.isnull().sum())
    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()
