import pandas as pd

from ml.config import RAW_DATASET, CLEAN_DATASET


def load_dataset():
    """
    Load the raw phishing dataset.
    """

    print("\nLoading dataset...")

    df = pd.read_excel(RAW_DATASET)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df

def validate_dataset(df):
    """
    Validate that the dataset contains the required columns.
    """

    print("\nValidating dataset...")

    required_columns = ["URL", "label"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset validation passed.")

def main():
    """
    Main preprocessing pipeline.
    """

    print("=" * 60)
    print("QR Phishing Detection - Dataset Preprocessing")
    print("=" * 60)

    df = load_dataset()

    validate_dataset(df)


if __name__ == "__main__":
    main()