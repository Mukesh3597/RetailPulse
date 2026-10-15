from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

FEATURES = [
    "Total_Orders",
    "Total_Spending",
    "Total_Quantity",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURES if col not in customer_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model = load_model()
    result = customer_data.copy()

    probabilities = model.predict_proba(result[FEATURES])[:, 1]
    result["Churn_Risk_Score"] = probabilities.round(4)
    result["Predicted_Churn"] = (probabilities >= 0.5).astype(int)

    return result
