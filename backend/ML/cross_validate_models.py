import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from ml.config import (
    FEATURE_DATASET,
    FEATURE_NAMES,
    RANDOM_STATE,
)


def main():

    print("=" * 60)
    print("QR Phishing Detection - 5-Fold Cross-Validation")
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
    # Create cross-validation strategy
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
    # Evaluate Logistic Regression
    # --------------------------------------------------

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X,
        y,
        cv,
    )

    # --------------------------------------------------
    # Evaluate Random Forest
    # --------------------------------------------------

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X,
        y,
        cv,
    )

    # --------------------------------------------------
    # Display comparison
    # --------------------------------------------------

    display_comparison(
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

def phishing_roc_auc(estimator, X, y):
    phishing_index = list(estimator.classes_).index(0)

    phishing_probability = estimator.predict_proba(X)[
        :, phishing_index
    ]

    phishing_actual = (y == 0).astype(int)

    return roc_auc_score(
        phishing_actual,
        phishing_probability,
    )

def evaluate_model(
    model_name,
    model,
    X,
    y,
    cv,
):

    print("\n")
    print("=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            pos_label=0,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            pos_label=0,
            zero_division=0,
        ),
        "roc_auc": phishing_roc_auc,
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

    print("\nFold Results:")

    for fold in range(5):

        print(f"\nFold {fold + 1}")

        print(
            f"Accuracy : "
            f"{results['test_accuracy'][fold]:.4f}"
        )

        print(
            f"Precision: "
            f"{results['test_precision'][fold]:.4f}"
        )

        print(
            f"Recall   : "
            f"{results['test_recall'][fold]:.4f}"
        )

        print(
            f"F1-Score : "
            f"{results['test_f1'][fold]:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{results['test_roc_auc'][fold]:.4f}"
        )

    print("\nMean ± Standard Deviation:")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    summary = {}

    for metric in metrics:

        test_scores = results[f"test_{metric}"]

        mean_score = test_scores.mean()
        std_score = test_scores.std()

        summary[metric] = (
            mean_score,
            std_score
        )

        print(
            f"{metric.capitalize():<10}: "
            f"{mean_score:.4f} ± {std_score:.4f}"
        )

    return {
        "name": model_name,
        "results": results,
        "summary": summary,
    }


def display_comparison(
    logistic_results,
    random_forest_results,
):

    print("\n")
    print("=" * 60)
    print("MODEL CROSS-VALIDATION COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Metric':<12}"
        f"{'Logistic Regression':<25}"
        f"{'Random Forest':<20}"
    )

    print("-" * 60)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in metrics:

        lr_mean, lr_std = (
            logistic_results["summary"][metric]
        )

        rf_mean, rf_std = (
            random_forest_results["summary"][metric]
        )

        print(
            f"{metric.capitalize():<12}"
            f"{lr_mean:.4f} ± {lr_std:.4f}"
            f"{'':<8}"
            f"{rf_mean:.4f} ± {rf_std:.4f}"
        )


if __name__ == "__main__":
    main()