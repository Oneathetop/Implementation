import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.config import (
    FEATURE_DATASET,
    FEATURE_NAMES,
    RANDOM_STATE,
)


def main():

    print("=" * 60)
    print("QR Phishing Detection - Overfitting Analysis")
    print("=" * 60)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------
    # Prepare features and labels
    # --------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nCross-validation strategy:")
    print("Folds: 5")
    print("Strategy: Stratified K-Fold")
    print("Shuffle: True")
    print(f"Random state: {RANDOM_STATE}")

    # --------------------------------------------------
    # Define models
    # --------------------------------------------------

    logistic_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # --------------------------------------------------
    # Analyze Logistic Regression
    # --------------------------------------------------

    logistic_results = analyze_model(
        "Logistic Regression",
        logistic_model,
        X,
        y,
        cv,
    )

    # --------------------------------------------------
    # Analyze Random Forest
    # --------------------------------------------------

    random_forest_results = analyze_model(
        "Random Forest",
        random_forest_model,
        X,
        y,
        cv,
    )

    # --------------------------------------------------
    # Compare overfitting
    # --------------------------------------------------

    compare_overfitting(
        logistic_results,
        random_forest_results,
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


def analyze_model(
    model_name,
    model,
    X,
    y,
    cv,
):

    print("\n")
    print("=" * 60)
    print(f"OVERFITTING ANALYSIS - {model_name}")
    print("=" * 60)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    summary = {}

    print("\nTraining vs Validation Performance:")

    for metric in metrics:

        train_scores = results[f"train_{metric}"]
        validation_scores = results[f"test_{metric}"]

        train_mean = train_scores.mean()
        validation_mean = validation_scores.mean()

        train_std = train_scores.std()
        validation_std = validation_scores.std()

        performance_gap = (
            train_mean - validation_mean
        )

        summary[metric] = {
            "train_mean": train_mean,
            "train_std": train_std,
            "validation_mean": validation_mean,
            "validation_std": validation_std,
            "gap": performance_gap,
        }

        print("\n" + metric.upper())

        print(
            f"Training   : "
            f"{train_mean:.4f} ± {train_std:.4f}"
        )

        print(
            f"Validation : "
            f"{validation_mean:.4f} ± {validation_std:.4f}"
        )

        print(
            f"Gap        : "
            f"{performance_gap:.4f}"
        )

    return {
        "name": model_name,
        "summary": summary,
    }


def compare_overfitting(
    logistic_results,
    random_forest_results,
):

    print("\n")
    print("=" * 60)
    print("OVERFITTING GAP COMPARISON")
    print("=" * 60)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    print(
        f"\n{'Metric':<12}"
        f"{'Logistic Regression':<25}"
        f"{'Random Forest':<20}"
    )

    print("-" * 60)

    for metric in metrics:

        lr_gap = (
            logistic_results["summary"]
            [metric]["gap"]
        )

        rf_gap = (
            random_forest_results["summary"]
            [metric]["gap"]
        )

        print(
            f"{metric.capitalize():<12}"
            f"{lr_gap:.4f}"
            f"{'':<20}"
            f"{rf_gap:.4f}"
        )

    print("\nInterpretation:")
    print(
        "A small training-validation gap indicates "
        "similar performance on training and validation data."
    )

    print(
        "A large positive gap indicates that the model "
        "performs substantially better on training data "
        "than validation data."
    )

    print(
        "The gap must be interpreted together with the "
        "cross-validation results and later domain-grouped evaluation."
    )


if __name__ == "__main__":
    main()