import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from ml.config import (
    FEATURE_DATASET,
    FEATURE_NAMES,
    TEST_SIZE,
    RANDOM_STATE,
)


def main():

    print("=" * 60)
    print("QR Phishing Detection - Random Forest Evaluation")
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

    evaluate_random_forest(
        random_forest_model,
        X_test,
        y_test,
    )


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


def evaluate_random_forest(model, X_test, y_test):

    print("\nEvaluating Random Forest...")

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing",
        ],
        zero_division=0,
    )

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()