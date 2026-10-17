"""
RetailPulse - Customer Churn Risk Model
Run from the repository root:
    python src/churn/churn_model.py

Important:
The current target is treated as a supplied label. If it was created using
Recency_Days > 90, that is a proxy definition of churn, not observed future churn.
Recency_Days and Last_Purchase are excluded from model inputs to avoid direct
target leakage. Model metrics describe this dataset/split only.
"""
from pathlib import Path
import json
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "customer_churn_data.csv"
SEGMENT_FILE = ROOT / "data" / "processed" / "customer_segments.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "customer_churn_predictions.csv"
METRICS_FILE = ROOT / "reports" / "churn_metrics.json"
MODEL_FILE = ROOT / "models" / "churn_model.pkl"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Copy the team's customer_churn_data.csv "
            "into data/processed/."
        )
    df = pd.read_csv(INPUT_FILE)
    # Standardize a few common column-name variants.
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in {"customer_id", "customerid"}:
            rename[col] = "Customer ID"
        elif key in {"churn", "churn_label"}:
            rename[col] = "Churn"
        elif key in {"total_orders", "frequency"}:
            rename[col] = "Total_Orders"
        elif key in {"total_spending", "monetary"}:
            rename[col] = "Total_Spending"
        elif key in {"total_quantity"}:
            rename[col] = "Total_Quantity"
        elif key in {"recency_days", "recency"}:
            rename[col] = "Recency_Days"
        elif key in {"last_purchase"}:
            rename[col] = "Last_Purchase"
    return df.rename(columns=rename)


def main():
    for path in [OUTPUT_FILE.parent, METRICS_FILE.parent, MODEL_FILE.parent]:
        path.mkdir(parents=True, exist_ok=True)
    df = load_data()
    if "Churn" not in df.columns:
        raise ValueError("Input file must include a Churn target column.")
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    df = df[df["Churn"].isin([0, 1])].copy()
    df["Churn"] = df["Churn"].astype(int)
    if df["Churn"].nunique() != 2:
        raise ValueError("Churn target must contain both 0 and 1 classes.")

    # Use customer behavior fields, but exclude direct rule columns.
    candidate_features = ["Total_Orders", "Total_Spending", "Total_Quantity"]
    feature_cols = [c for c in candidate_features if c in df.columns]
    if not feature_cols:
        raise ValueError(f"No supported features found. Available columns: {df.columns.tolist()}")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["Churn"]
    if y.value_counts().min() < 2:
        raise ValueError("Need at least two examples in each churn class for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "target": "Churn",
        "target_definition": "Supplied proxy label; verify whether it means no purchase in the last 90 days.",
        "features_used": feature_cols,
        "excluded_columns": [c for c in ["Recency_Days", "Last_Purchase"] if c in df.columns],
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "churn_class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "confusion_matrix_labels_order": [0, 1],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "note": "Metrics are from a random stratified holdout split. They are not proof of future churn prediction quality; validate on a time-based split and confirmed future outcomes before production use."
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Fit the final model on all labeled rows after evaluation, for downstream scoring.
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)

    all_prob = model.predict_proba(X)[:, 1]
    predictions = df[["Customer ID"]].copy() if "Customer ID" in df.columns else pd.DataFrame(index=df.index)
    predictions["Actual_Churn_Label"] = y
    predictions["Churn_Risk_Score"] = all_prob.round(4)
    predictions["Predicted_Churn"] = (all_prob >= 0.5).astype(int)
    predictions["Risk_Level"] = pd.cut(
        all_prob, bins=[-0.001, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"], include_lowest=True
    ).astype(str)
    predictions.to_csv(OUTPUT_FILE, index=False)

    print("Churn model evaluation (holdout set):")
    for key in ["accuracy", "precision", "recall", "roc_auc"]:
        print(f"{key.title()}: {metrics[key]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved predictions: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
