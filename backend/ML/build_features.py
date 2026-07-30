import pandas as pd
from pathlib import Path

from ml.config import (
    CLEAN_DATASET,
    FEATURE_DATASET,
    FEATURE_NAMES,
)

from ml.feature_extractor import extract_features

def main():
    """
    Build the machine learning feature dataset.
    """

    print("=" * 60)
    print("QR Phishing Detection - Feature Engineering")
    print("=" * 60)

    df = load_dataset()
    validate_dataset(df)
    feature_df = build_feature_dataset(df)
    validate_feature_dataset(feature_df)
    save_feature_dataset(feature_df)

def load_dataset():
    """
    Load the cleaned dataset.
    """

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(CLEAN_DATASET)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df

def validate_dataset(df):
    """
    Validate the cleaned dataset before feature extraction.
    """

    print("\nValidating dataset...")

    required_columns = ["URL", "label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("Dataset validation passed.")

def build_feature_dataset(df):
    """
    Extract lexical features for every URL.
    """

    print("\nExtracting features...")

    feature_rows = []

    for _, row in df.iterrows():

        features = extract_features(row["URL"])

        features["label"] = row["label"]

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    print("Feature extraction completed.")

    print(f"Rows: {len(feature_df)}")
    print(f"Columns: {len(feature_df.columns)}")

    return feature_df

def validate_feature_dataset(feature_df):
    """
    Validate the generated feature dataset.
    """

    print("\nValidating feature dataset...")

    expected_columns = FEATURE_NAMES + ["label"]

    missing = [
        column
        for column in expected_columns
        if column not in feature_df.columns
    ]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print("Feature dataset validation passed.")

def save_feature_dataset(feature_df):
    """
    Save the engineered feature dataset.
    """

    print("\nSaving feature dataset...")

    FEATURE_DATASET.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(FEATURE_DATASET, index=False)

    print("Feature dataset saved to:")
    print(FEATURE_DATASET.resolve())

if __name__ == "__main__":
    main()