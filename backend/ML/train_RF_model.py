import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.config import (
    FEATURE_DATASET,
    FEATURE_NAMES,
    TEST_SIZE,
    RANDOM_STATE,
)

# Set model target save path 
MODEL_SAVE_PATH = "models/random_forest_model.joblib"

def main():

    print("=" * 60)
    print("QR Phishing Detection - Random Forest Training")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    random_forest_model = train_random_forest(
        X_train,
        y_train,
    )

    save_model(random_forest_model, MODEL_SAVE_PATH)

def load_dataset():

    print("\nLoading feature dataset...")

    df = pd.read_csv(FEATURE_DATASET)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def prepare_data(df):

    print("\nPreparing features and labels...")

    X = df[FEATURE_NAMES]

    y = df["label"]

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")

    return X, y


def split_dataset(X, y):

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Dataset split completed.")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Random Forest training completed.")

    return model

def save_model(model, filepath):
    print(f"\nSaving trained model to {filepath}...") 
    joblib.dump(model, filepath) 
    print("Model saved successfully.")

if __name__ == "__main__":
    main()