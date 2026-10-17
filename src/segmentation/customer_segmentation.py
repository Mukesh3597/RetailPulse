"""
RetailPulse - Customer Segmentation using RFM + K-Means
Run from the repository root:
    python src/segmentation/customer_segmentation.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
EXISTING_SEGMENTS = ROOT / "data" / "processed" / "customer_segments.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "customer_segments.csv"
SUMMARY_FILE = ROOT / "reports" / "customer_segment_summary.csv"
RFM_FILE = ROOT / "data" / "processed" / "customer_rfm.csv"
N_CLUSTERS = 6


def load_transactions():
    """Prefer the team's existing cleaned transaction data when available."""
    possible = [
        ROOT / "data" / "processed" / "cleaned_transactions.csv",
        ROOT / "data" / "processed" / "RetailPulse_Cleaned_Dataset.csv",
        ROOT / "data" / "cleaned_data.csv",
    ]
    for path in possible:
        if path.exists():
            df = pd.read_csv(path)
            cols = {str(c).strip().lower(): c for c in df.columns}
            if {"customer id", "invoicedate", "quantity", "price"}.issubset(cols):
                df = df.rename(columns={
                    cols["customer id"]: "Customer ID",
                    cols["invoicedate"]: "InvoiceDate",
                    cols["quantity"]: "Quantity",
                    cols["price"]: "Price",
                })
                if "Invoice" not in df.columns:
                    df["Invoice"] = df.index.astype(str)
                return df
    if RAW_FILE.exists():
        sheets = pd.read_excel(RAW_FILE, sheet_name=None)
        return pd.concat(sheets.values(), ignore_index=True)
    raise FileNotFoundError(
        "No transaction file found. Put online_retail_II.xlsx in data/raw/ "
        "or save Anu's cleaned transaction-level data as data/processed/cleaned_transactions.csv."
    )


def make_rfm(transactions):
    """Create one row per customer from transaction-level data."""
    df = transactions.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Customer ID", "InvoiceDate", "Quantity", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Customer ID", "InvoiceDate", "Quantity", "Price"])
    # Exclude cancellations/returns and invalid transaction values.
    if "Invoice" in df.columns:
        df = df[~df["Invoice"].astype(str).str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    if df.empty:
        raise ValueError("No valid transactions remain after cleaning.")

    snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Last_Purchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique") if "Invoice" in df.columns else ("InvoiceDate", "count"),
        Monetary=("TotalPrice", "sum"),
        Total_Quantity=("Quantity", "sum"),
    ).reset_index()
    rfm["Recency"] = (snapshot_date - rfm["Last_Purchase"].dt.normalize()).dt.days
    rfm = rfm.rename(columns={"Customer ID": "CustomerID"})
    rfm["CustomerID"] = rfm["CustomerID"].astype("int64")
    return rfm, snapshot_date




def name_segments(summary):
    """Assign business-friendly names to the six clusters."""

    summary = summary.copy()

    # Rank clusters by their customer behavior.
    summary["R_Rank"] = summary["Avg_Recency"].rank(
        ascending=True, method="min"
    )
    summary["F_Rank"] = summary["Avg_Frequency"].rank(
        ascending=False, method="min"
    )
    summary["M_Rank"] = summary["Avg_Monetary"].rank(
        ascending=False, method="min"
    )

    names = {}

    for cluster, row in summary.iterrows():
        recency = row["Avg_Recency"]
        frequency = row["Avg_Frequency"]
        spending = row["Avg_Monetary"]

        if frequency >= 20 and spending >= 10000:
            name = "Champions"

        elif frequency >= 7 and spending >= 3000:
            name = "Loyal High-Value Customers"

        elif recency <= 60 and frequency < 5:
            name = "Recent Low-Frequency Customers"

        elif recency > 180 and spending >= 1000:
            name = "At-Risk Valuable Customers"

        elif recency > 180:
            name = "Inactive Customers"

        elif frequency >= 3 and spending >= 1000:
            name = "Regular Customers"

        else:
            name = "Occasional Customers"

        names[cluster] = name

    return names




def main():
    for path in [OUTPUT_FILE.parent, SUMMARY_FILE.parent]:
        path.mkdir(parents=True, exist_ok=True)
    transactions = load_transactions()
    rfm, snapshot_date = make_rfm(transactions)

    # Log transform reduces the effect of very large customer values.
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Frequency"] = features["Frequency"].clip(lower=1)
    features["Monetary"] = features["Monetary"].clip(lower=0)
    transformed = features.copy()
    transformed["Frequency"] = np.log1p(transformed["Frequency"])
    transformed["Monetary"] = np.log1p(transformed["Monetary"])
    scaled = StandardScaler().fit_transform(transformed)

    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    rfm["Segment"] = model.fit_predict(scaled)
    summary = rfm.groupby("Segment").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).round(2)
    names = name_segments(summary)
    rfm["Segment_Name"] = rfm["Segment"].map(names)
    summary["Segment_Name"] = summary.index.map(names)
    summary = summary.reset_index()

    rfm.to_csv(RFM_FILE, index=False)
    rfm.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Customers segmented: {len(rfm):,}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
