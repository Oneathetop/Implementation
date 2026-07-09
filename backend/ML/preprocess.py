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


def main():
    """
    Main preprocessing pipeline.
    """

    print("=" * 60)
    print("QR Phishing Detection - Dataset Preprocessing")
    print("=" * 60)

    df = load_dataset()


if __name__ == "__main__":
    main()