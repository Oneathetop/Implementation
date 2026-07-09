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

def select_required_columns(df):
    """
    Keep only the required columns for feature engineering.
    """

    print("\nSelecting required columns...")

    clean_df = df[["URL", "label"]].copy()

    print(f"Selected columns: {list(clean_df.columns)}")

    return clean_df

def remove_missing_values(df):
    """
    Remove rows with missing values.
    """

    print("\nRemoving missing values...")

    rows_before = len(df)

    df = df.dropna()

    rows_after = len(df)

    print(f"Removed {rows_before - rows_after} rows.")

    return df
def remove_duplicates(df):
    """
    Remove duplicate URLs.
    """

    print("\nRemoving duplicate URLs...")

    rows_before = len(df)

    df = df.drop_duplicates(subset=["URL"])

    rows_after = len(df)

    print(f"Removed {rows_before - rows_after} duplicate URLs.")

    return df


def save_clean_dataset(df):
    """
    Save the cleaned dataset as CSV.
    """

    print("\nSaving cleaned dataset...")

    df.to_csv(CLEAN_DATASET, index=False)

    print(f"Dataset saved to:\n{CLEAN_DATASET}")


def print_statistics(df):
    """
    Display final dataset statistics.
    """

    print("\n" + "=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)

    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    print("\nClass Distribution:")

    print(df["label"].value_counts())

    print("\nPreview:")

    print(df.head())
    
def main():
    """
    Main preprocessing pipeline.
    """

    print("=" * 60)
    print("QR Phishing Detection - Dataset Preprocessing")
    print("=" * 60)

    df = load_dataset()

    validate_dataset(df)

    clean_df = select_required_columns(df)

    clean_df = remove_missing_values(clean_df)

    clean_df = remove_duplicates(clean_df)

    save_clean_dataset(clean_df)

    print_statistics(clean_df)

if __name__ == "__main__":
    main()