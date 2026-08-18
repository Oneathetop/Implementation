import pandas as pd
import numpy as np
import tldextract

from sklearn.model_selection import GroupShuffleSplit
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
)

from ml.config import (
    CLEAN_DATASET,
    FEATURE_DATASET,
    FEATURE_NAMES,
    TEST_SIZE,
    RANDOM_STATE,
)


# ============================================================
# ROOT DOMAIN
# ============================================================

def extract_root_domain(url):

    extracted = tldextract.extract(url)

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"

    if extracted.domain:
        return extracted.domain

    return "unknown"


# ============================================================
# DATA LOADING
# ============================================================

def load_feature_dataset():

    print("\nLoading feature dataset...")

    df = pd.read_csv(FEATURE_DATASET)

    print("Feature dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def load_url_dataset():

    print("\nLoading cleaned URL dataset...")

    df = pd.read_csv(CLEAN_DATASET)

    print("Cleaned URL dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def verify_alignment(feature_df, url_df):

    print("\nVerifying dataset alignment...")

    if len(feature_df) != len(url_df):

        raise ValueError(
            "Feature dataset and URL dataset "
            "have different numbers of rows."
        )

    print("PASS: Row counts match.")

    if not feature_df["label"].equals(
        url_df["label"]
    ):

        mismatch_count = (
            feature_df["label"]
            != url_df["label"]
        ).sum()

        raise ValueError(
            f"Label alignment failure: "
            f"{mismatch_count} rows differ."
        )

    print("PASS: Labels are aligned row-by-row.")

    if "URL" not in url_df.columns:

        raise ValueError(
            "Cleaned URL dataset does not contain "
            "the required URL column."
        )

    print(
        "PASS: Original URLs available in "
        "cleaned URL dataset."
    )

    print(
        "PASS: Feature dataset and cleaned URL dataset "
        "are aligned by row count and label."
    )


# ============================================================
# CREATE ROOT-DOMAIN HOLDOUT
# ============================================================

def create_robustness_holdout(
    url_df,
    test_size=TEST_SIZE,
):

    groups = url_df["URL"].apply(
        extract_root_domain
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            url_df,
            url_df["label"],
            groups=groups,
        )
    )

    train_groups = set(
        groups.iloc[train_indices]
    )

    test_groups = set(
        groups.iloc[test_indices]
    )

    overlap = (
        train_groups
        & test_groups
    )

    print("\n")
    print("=" * 60)
    print("ROBUSTNESS HOLDOUT CHECK")
    print("=" * 60)

    print(
        f"Training URLs: {len(train_indices)}"
    )

    print(
        f"Testing URLs: {len(test_indices)}"
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
        f"Root-domain overlap: {len(overlap)}"
    )

    if overlap:

        raise ValueError(
            "Root-domain leakage detected "
            "in robustness holdout."
        )

    print(
        "PASS: Robustness holdout contains "
        "unseen root domains."
    )

    return (
        train_indices,
        test_indices,
        groups,
    )


# ============================================================
# MODEL CREATION
# ============================================================

def create_logistic_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def create_random_forest_model():

    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_from_url(url):

    extracted = tldextract.extract(url)

    domain = extracted.domain or ""
    suffix = extracted.suffix or ""
    subdomain = extracted.subdomain or ""

    domain_full = (
        f"{domain}.{suffix}"
        if domain and suffix
        else domain
    )

    url_string = str(url)

    features = {

        "url_length":
            len(url_string),

        "https":
            int(url_string.lower().startswith("https://")),

        "domain_length":
            len(domain_full),

        "path_length":
            len(
                url_string.split(
                    domain_full,
                    1
                )[-1]
            ) if domain_full else 0,

        "dot_count":
            url_string.count("."),

        "digit_count":
            sum(
                char.isdigit()
                for char in url_string
            ),

        "hyphen_count":
            url_string.count("-"),

        "special_character_count":
            sum(
                not char.isalnum()
                and char not in "/.:_-"
                for char in url_string
            ),

        "ip_address":
            int(
                any(
                    part.isdigit()
                    for part in domain.split(".")
                )
                and domain.count(".") == 3
            ),

        "subdomain_count":
            0
            if not subdomain
            else len(
                subdomain.split(".")
            ),
    }

    return features


# ============================================================
# FEATURE ALIGNMENT
# ============================================================

def create_feature_vector(url):

    features = extract_features_from_url(url)

    vector = []

    for feature_name in FEATURE_NAMES:

        vector.append(
            features.get(
                feature_name,
                0
            )
        )

    return vector


# ============================================================
# URL PERTURBATIONS
# ============================================================

def add_digits(url):

    return url + "123"


def add_https(url):

    if url.lower().startswith("https://"):
        return None

    if url.lower().startswith("http://"):
        return (
            "https://"
            + url[7:]
        )

    return "https://" + url


def add_special_characters(url):

    return url + "?&=%"


def add_suspicious_query(url):

    separator = "&" if "?" in url else "?"

    return (
        url
        + separator
        + "login=verify_account"
    )


def add_suspicious_subdomain(url):

    extracted = tldextract.extract(url)

    if not extracted.domain:
        return None

    root_domain = (
        f"{extracted.domain}.{extracted.suffix}"
        if extracted.suffix
        else extracted.domain
    )

    return (
        "secure-login."
        + root_domain
    )


def remove_https(url):

    if url.lower().startswith("https://"):

        return "http://" + url[8:]

    return None


PERTURBATIONS = {

    "add_digits":
        add_digits,

    "add_https":
        add_https,

    "add_special_characters":
        add_special_characters,

    "add_suspicious_query":
        add_suspicious_query,

    "add_suspicious_subdomain":
        add_suspicious_subdomain,

    "remove_https":
        remove_https,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_probability,
            ),
    }


# ============================================================
# ROBUSTNESS EVALUATION
# ============================================================

def evaluate_robustness(
    model_name,
    model,
    urls,
    labels,
):

    print("\n")
    print("=" * 60)
    print(
        f"ROBUSTNESS TEST - {model_name.upper()}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_features = np.array([
        create_feature_vector(url)
        for url in urls
    ])

    baseline_predictions = model.predict(
        baseline_features
    )

    baseline_probabilities = (
        model.predict_proba(
            baseline_features
        )[:, 1]
    )

    baseline_metrics = calculate_metrics(
        labels,
        baseline_predictions,
        baseline_probabilities,
    )

    print("\nBaseline performance:")

    for metric, value in baseline_metrics.items():

        print(
            f"{metric.upper():<10}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Perturbation tests
    # --------------------------------------------------------

    all_results = []

    for perturbation_name, perturbation_function in PERTURBATIONS.items():

        print("\n")
        print("-" * 60)
        print(
            f"Testing perturbation: "
            f"{perturbation_name}"
        )
        print("-" * 60)

        test_urls = []
        test_labels = []
        original_predictions = []
        original_probabilities = []

        # ----------------------------------------------------
        # Generate valid perturbations
        # ----------------------------------------------------

        for url, label in zip(
            urls,
            labels,
        ):

            perturbed_url = perturbation_function(
                url
            )

            if perturbed_url is None:
                continue

            test_urls.append(
                perturbed_url
            )

            test_labels.append(
                label
            )

            original_predictions.append(
                model.predict(
                    np.array([
                        create_feature_vector(url)
                    ])
                )[0]
            )

            original_probabilities.append(
                model.predict_proba(
                    np.array([
                        create_feature_vector(url)
                    ])
                )[0, 1]
            )

        if not test_urls:

            print(
                "No valid perturbations "
                "generated."
            )

            continue

        # ----------------------------------------------------
        # Extract perturbed features
        # ----------------------------------------------------

        perturbed_features = np.array([
            create_feature_vector(url)
            for url in test_urls
        ])

        perturbed_predictions = model.predict(
            perturbed_features
        )

        perturbed_probabilities = (
            model.predict_proba(
                perturbed_features
            )[:, 1]
        )

        test_labels = np.array(
            test_labels
        )

        original_predictions = np.array(
            original_predictions
        )

        original_probabilities = np.array(
            original_probabilities
        )

        # ----------------------------------------------------
        # Correctness
        # ----------------------------------------------------

        original_correct = (
            original_predictions
            == test_labels
        )

        perturbed_correct = (
            perturbed_predictions
            == test_labels
        )

        prediction_changed = (
            original_predictions
            != perturbed_predictions
        )

        correct_to_incorrect = (
            original_correct
            & ~perturbed_correct
        )

        incorrect_to_correct = (
            ~original_correct
            & perturbed_correct
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        perturbed_metrics = calculate_metrics(
            test_labels,
            perturbed_predictions,
            perturbed_probabilities,
        )

        prediction_change_rate = (
            prediction_changed.mean()
        )

        robustness_failure_rate = (
            correct_to_incorrect.mean()
        )

        recovery_rate = (
            incorrect_to_correct.mean()
        )

        probability_change = (
            perturbed_probabilities
            - original_probabilities
        )

        mean_probability_change = (
            probability_change.mean()
        )

        mean_absolute_probability_change = (
            np.abs(
                probability_change
            ).mean()
        )

        accuracy_degradation = (
            baseline_metrics["accuracy"]
            - perturbed_metrics["accuracy"]
        )

        f1_degradation = (
            baseline_metrics["f1"]
            - perturbed_metrics["f1"]
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Tests: "
            f"{len(test_urls)}"
        )

        print(
            f"Prediction change rate: "
            f"{prediction_change_rate:.4f}"
        )

        print(
            f"Correct -> Incorrect rate: "
            f"{robustness_failure_rate:.4f}"
        )

        print(
            f"Incorrect -> Correct rate: "
            f"{recovery_rate:.4f}"
        )

        print(
            f"Mean probability change: "
            f"{mean_probability_change:.4f}"
        )

        print(
            f"Mean absolute probability change: "
            f"{mean_absolute_probability_change:.4f}"
        )

        print(
            f"Accuracy degradation: "
            f"{accuracy_degradation:.4f}"
        )

        print(
            f"F1 degradation: "
            f"{f1_degradation:.4f}"
        )

        print("\nPerturbed metrics:")

        for metric, value in perturbed_metrics.items():

            print(
                f"{metric.upper():<10}: "
                f"{value:.4f}"
            )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        all_results.append({

            "perturbation":
                perturbation_name,

            "tests":
                len(test_urls),

            "prediction_change_rate":
                prediction_change_rate,

            "robustness_failure_rate":
                robustness_failure_rate,

            "recovery_rate":
                recovery_rate,

            "mean_probability_change":
                mean_probability_change,

            "mean_absolute_probability_change":
                mean_absolute_probability_change,

            "accuracy_degradation":
                accuracy_degradation,

            "f1_degradation":
                f1_degradation,

            "accuracy":
                perturbed_metrics["accuracy"],

            "precision":
                perturbed_metrics["precision"],

            "recall":
                perturbed_metrics["recall"],

            "f1":
                perturbed_metrics["f1"],

            "roc_auc":
                perturbed_metrics["roc_auc"],
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print("\n")
    print("=" * 60)
    print(
        f"{model_name.upper()} - "
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 60)

    if results_df.empty:

        print(
            "No robustness results available."
        )

        return results_df

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "QR Phishing Detection - Robustness Testing"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    feature_df = load_feature_dataset()

    url_df = load_url_dataset()

    verify_alignment(
        feature_df,
        url_df,
    )

    # --------------------------------------------------------
    # Create unseen-root-domain holdout
    # --------------------------------------------------------

    (
        train_indices,
        test_indices,
        groups,
    ) = create_robustness_holdout(
        url_df
    )

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    X_train = feature_df[
        FEATURE_NAMES
    ].iloc[train_indices]

    y_train = feature_df[
        "label"
    ].iloc[train_indices]

    # --------------------------------------------------------
    # Prepare robustness test data
    # --------------------------------------------------------

    robustness_url_df = url_df.iloc[
        test_indices
    ].copy()

    # --------------------------------------------------------
    # Sample 200 URLs
    # --------------------------------------------------------

    sample_size = min(
        200,
        len(robustness_url_df)
    )

    robustness_sample = (
        robustness_url_df
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"\nRobustness URLs sampled: "
        f"{len(robustness_sample)}"
    )

    urls = (
        robustness_sample["URL"]
        .tolist()
    )

    labels = (
        robustness_sample["label"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = (
        create_logistic_model()
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    print(
        "Logistic Regression training "
        "completed."
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest_model = (
        create_random_forest_model()
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    print(
        "Random Forest training "
        "completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_robustness(
        "Logistic Regression",
        logistic_model,
        urls,
        labels,
    )

    evaluate_robustness(
        "Random Forest",
        random_forest_model,
        urls,
        labels,
    )

    print("\n")
    print("=" * 60)
    print(
        "ROBUSTNESS TEST COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()