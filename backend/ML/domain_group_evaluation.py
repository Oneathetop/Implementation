import pandas as pd
import tldextract

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    CLEAN_DATASET,
    FEATURE_DATASET,
    FEATURE_NAMES,
    RANDOM_STATE,
)


# --------------------------------------------------
# Root Domain Extraction
# --------------------------------------------------

def extract_root_domain(url):

    extracted = tldextract.extract(url)

    if extracted.domain and extracted.suffix:

        return (
            f"{extracted.domain}."
            f"{extracted.suffix}"
        )

    if extracted.domain:

        return extracted.domain

    return "unknown"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("QR Phishing Detection - Root-Domain Grouped Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------

    feature_df = load_feature_dataset()

    # --------------------------------------------------
    # Load original URL dataset
    # --------------------------------------------------

    url_df = load_url_dataset()

    # --------------------------------------------------
    # Verify row alignment
    # --------------------------------------------------

    verify_row_alignment(
        feature_df,
        url_df,
    )

    # --------------------------------------------------
    # Extract root domains
    # --------------------------------------------------

    groups = create_domain_groups(
        url_df
    )

    # --------------------------------------------------
    # Prepare features and labels
    # --------------------------------------------------

    X, y = prepare_data(
        feature_df
    )

    # --------------------------------------------------
    # Group statistics
    # --------------------------------------------------

    display_group_statistics(
        groups,
        y,
    )

    # --------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------
    # Evaluate Logistic Regression
    # --------------------------------------------------

    evaluate_model(
        "Logistic Regression",
        create_logistic_model(),
        X,
        y,
        groups,
        cv,
    )

    # --------------------------------------------------
    # Evaluate Random Forest
    # --------------------------------------------------

    evaluate_model(
        "Random Forest",
        create_random_forest_model(),
        X,
        y,
        groups,
        cv,
    )


# --------------------------------------------------
# Dataset Loading
# --------------------------------------------------

def load_feature_dataset():

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        FEATURE_DATASET
    )

    print("Feature dataset loaded successfully.")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


def load_url_dataset():

    print("\nLoading cleaned URL dataset...")

    df = pd.read_csv(
        CLEAN_DATASET
    )

    print("Cleaned URL dataset loaded successfully.")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# --------------------------------------------------
# Dataset Validation
# --------------------------------------------------

def verify_row_alignment(feature_df, url_df):

    print("\nVerifying dataset alignment...")

    # 1. Row count check
    if len(feature_df) != len(url_df):
        raise ValueError(
            "Feature dataset and URL dataset "
            "have different numbers of rows."
        )

    print("PASS: Row counts match.")

    # 2. Label alignment check
    if not feature_df["label"].equals(url_df["label"]):
        mismatch_count = (
            feature_df["label"] != url_df["label"]
        ).sum()

        raise ValueError(
            f"Label alignment failure: "
            f"{mismatch_count} rows have different labels."
        )

    print("PASS: Labels are aligned row-by-row.")

    print(
        "PASS: Feature dataset and cleaned URL dataset "
        "are aligned for root-domain grouping."
    )


# --------------------------------------------------
# Root Domain Groups
# --------------------------------------------------

def create_domain_groups(
    url_df,
):

    print("\nExtracting root domains...")

    groups = url_df["URL"].apply(
        extract_root_domain
    )

    print(
        f"Unique root domains: "
        f"{groups.nunique()}"
    )

    print(
    f"URLs with valid root domain: "
    f"{(groups != 'unknown').sum()}"
)

    print(
        f"URLs with missing/unknown domain: "
        f"{(groups == 'unknown').sum()}"
    )

    return groups


# --------------------------------------------------
# Prepare Data
# --------------------------------------------------

def prepare_data(
    feature_df,
):

    print("\nPreparing features and labels...")

    X = feature_df[
        FEATURE_NAMES
    ]

    y = feature_df[
        "label"
    ]

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Samples: {X.shape[0]}"
    )

    return X, y


# --------------------------------------------------
# Group Statistics
# --------------------------------------------------

def display_group_statistics(
    groups,
    y,
):

    print("\n")
    print("-" * 60)
    print("ROOT-DOMAIN GROUP STATISTICS")
    print("-" * 60)

    print(
        f"Total URLs: {len(groups)}"
    )

    print(
        f"Unique root domains: "
        f"{groups.nunique()}"
    )

    domain_counts = (
        groups
        .value_counts()
        .head(10)
    )

    print("\nTop 10 root domains:")

    print(domain_counts)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

def create_logistic_model():

    return Pipeline([
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


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

def create_random_forest_model():

    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X,
    y,
    groups,
    cv,
):

    print("\n")
    print("=" * 60)
    print(
        f"ROOT-DOMAIN EVALUATION - "
        f"{model_name}"
    )
    print("=" * 60)

    fold_results = []

    for fold, (
        train_indices,
        test_indices
    ) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        train_groups = set(
            groups.iloc[
                train_indices
            ]
        )

        test_groups = set(
            groups.iloc[
                test_indices
            ]
        )

        overlap = (
            train_groups
            & test_groups
        )

        if overlap:

            raise ValueError(
                f"Domain leakage detected "
                f"in fold {fold}: "
                f"{len(overlap)} overlapping "
                f"root domains."
            )

        print("\n" + "-" * 60)
        print(
            f"Fold {fold}"
        )
        print("-" * 60)

        print(
            f"Training URLs: "
            f"{len(X_train)}"
        )

        print(
            f"Testing URLs: "
            f"{len(X_test)}"
        )

        print(
            f"Training root domains: "
            f"{len(train_groups)}"
        )

        print(
            f"Testing root domains: "
            f"{len(test_groups)}"
        )

        print(
            "Root-domain overlap: 0"
        )

        # ------------------------------------------
        # Train
        # ------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ------------------------------------------
        # Predict
        # ------------------------------------------

        y_pred = model.predict(
            X_test
        )

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1-Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(matrix)

        fold_results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        })

    # --------------------------------------------------
    # Aggregate Results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        f"{model_name} - "
        f"ROOT-DOMAIN GROUPED RESULTS"
    )
    print("=" * 60)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in metrics:

        values = [
            result[metric]
            for result in fold_results
        ]

        mean_value = (
            sum(values)
            / len(values)
        )

        variance = sum(
            (
                value - mean_value
            ) ** 2
            for value in values
        ) / len(values)

        std_value = (
            variance ** 0.5
        )

        print(
            f"{metric.upper():<10}: "
            f"{mean_value:.4f} "
            f"± {std_value:.4f}"
        )


if __name__ == "__main__":
    main()