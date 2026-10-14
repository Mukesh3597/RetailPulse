from pathlib import Path
import pandas as pd


def load_retail_data(file_path):
    """
    Load both sheets of the Online Retail II dataset
    and combine them into one DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df_2009 = pd.read_excel(file_path, sheet_name="Year 2009-2010")

    df_2010 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

    df = pd.concat([df_2009, df_2010], ignore_index=True)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    file_path = project_root / "data" / "raw" / "online_retail_II.xlsx"

    df = load_retail_data(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
