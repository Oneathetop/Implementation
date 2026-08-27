import pandas as pd
import tldextract
import numpy as np

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

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

from ml.feature_extractor import extract_features


# ============================================================
# CONSTANTS
# ============================================================

ROBUSTNESS_SAMPLE_SIZE = 200

ROBUSTNESS_RESULTS_FILE = (
    "datasets/processed/robustness_results.csv"
)


# ============================================================
# ROOT DOMAIN
# ============================================================

def extract_root_domain(url):
    """
    Extract the registrable/root domain from a URL.

    Examples:
        https://www.google.com
            -> google.com

        https://login.google.com
            -> google.com
    """

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

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if "label" not in feature_df.columns:
        raise ValueError(
            "Feature dataset does not contain 'label'."
        )

    if "label" not in url_df.columns:
        raise ValueError(
            "Cleaned URL dataset does not contain 'label'."
        )

    if "URL" not in url_df.columns:
        raise ValueError(
            "Cleaned URL dataset does not contain 'URL'."
        )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(feature_df) != len(url_df):

        raise ValueError(
            "Feature dataset and URL dataset "
            "have different numbers of rows."
        )

    print("PASS: Row counts match.")

    # --------------------------------------------------------
    # Label alignment
    # --------------------------------------------------------

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

    print(
        "PASS: Labels are aligned row-by-row."
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in feature_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Feature dataset is missing required "
            f"model features: {missing_features}"
        )

    print(
        "PASS: Feature dataset contains all "
        "required model features."
    )

    # --------------------------------------------------------
    # URL availability
    # --------------------------------------------------------

    print(
        "PASS: Original URLs available in "
        "cleaned URL dataset."
    )

    print(
        "PASS: Feature dataset and cleaned URL "
        "dataset are aligned by row count and label."
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
# FEATURE VECTOR CREATION
# ============================================================

def create_feature_vector(url):
    """
    Extract production features from a URL and return
    them in exactly the same order as FEATURE_NAMES.
    """

    features = extract_features(url)

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in features
    ]

    if missing_features:

        raise ValueError(
            "Feature extraction mismatch. "
            f"Missing features: {missing_features}"
        )

    extra_features = [
        feature
        for feature in features
        if feature not in FEATURE_NAMES
    ]

    if extra_features:

        raise ValueError(
            "Feature extraction mismatch. "
            f"Unexpected features: {extra_features}"
        )

    return [
        features[feature]
        for feature in FEATURE_NAMES
    ]


def create_feature_dataframe(urls):

    feature_rows = [
        create_feature_vector(url)
        for url in urls
    ]

    return pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
    )


# ============================================================
# PERTURBATION HELPERS
# ============================================================

def parse_url(url):
    """
    Parse URL while preserving its components.
    """

    parsed = urlsplit(url)

    return parsed


def rebuild_url(
    parsed,
    scheme=None,
    netloc=None,
    path=None,
    query=None,
    fragment=None,
):

    return urlunsplit((
        scheme if scheme is not None else parsed.scheme,
        netloc if netloc is not None else parsed.netloc,
        path if path is not None else parsed.path,
        query if query is not None else parsed.query,
        fragment if fragment is not None else parsed.fragment,
    ))


# ============================================================
# URL PERTURBATIONS
# ============================================================

def add_digits(url):
    """
    Add digits to the PATH rather than the domain.

    Example:
        https://example.com/login
        ->
        https://example.com/login123
    """

    parsed = parse_url(url)

    path = parsed.path

    if not path:
        path = "/"

    return rebuild_url(
        parsed,
        path=path + "123",
    )


def add_https(url):
    """
    Convert HTTP to HTTPS.

    Only HTTP URLs are eligible.

    HTTPS URLs are skipped because adding HTTPS to an
    already-HTTPS URL would produce no transformation.
    """

    parsed = parse_url(url)

    if parsed.scheme.lower() == "http":

        return rebuild_url(
            parsed,
            scheme="https",
        )

    return None


def add_special_characters(url):
    """
    Add URL-safe special characters through a legitimate
    query parameter.

    Example:
        https://example.com
        ->
        https://example.com/?ref=%26%3D%25
    """

    parsed = parse_url(url)

    existing_params = parse_qsl(
        parsed.query,
        keep_blank_values=True,
    )

    existing_params.append(
        ("ref", "&=%")
    )

    new_query = urlencode(
        existing_params
    )

    return rebuild_url(
        parsed,
        query=new_query,
    )


def add_suspicious_query(url):
    """
    Add a realistic suspicious-looking query parameter
    while preserving the original root domain.

    Example:
        https://example.com/login
        ->
        https://example.com/login?login=verify_account
    """

    parsed = parse_url(url)

    existing_params = parse_qsl(
        parsed.query,
        keep_blank_values=True,
    )

    existing_params.append(
        ("login", "verify_account")
    )

    new_query = urlencode(
        existing_params
    )

    return rebuild_url(
        parsed,
        query=new_query,
    )


def add_suspicious_subdomain(url):
    """
    Add a suspicious subdomain while preserving the
    original registrable/root domain.

    Example:
        https://example.com
        ->
        https://secure-login.example.com
    """

    parsed = parse_url(url)

    extracted = tldextract.extract(url)

    if not extracted.domain:
        return None

    if not extracted.suffix:

        return None

    root_domain = (
        f"{extracted.domain}.{extracted.suffix}"
    )

    suspicious_netloc = (
        "secure-login."
        + root_domain
    )

    return rebuild_url(
        parsed,
        netloc=suspicious_netloc,
    )


def remove_https(url):
    """
    Convert HTTPS to HTTP.

    Only HTTPS URLs are eligible.
    """

    parsed = parse_url(url)

    if parsed.scheme.lower() == "https":

        return rebuild_url(
            parsed,
            scheme="http",
        )

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
# PERTURBATION VALIDATION
# ============================================================

def validate_perturbation(
    original_url,
    perturbed_url,
    perturbation_name,
):
    """
    Validate that the perturbation actually changed the URL
    and, where appropriate, preserved the root domain.
    """

    if perturbed_url is None:
        return False

    if not isinstance(
        perturbed_url,
        str,
    ):
        return False

    if not perturbed_url:
        return False

    if perturbed_url == original_url:
        return False

    # --------------------------------------------------------
    # Root-domain preservation
    # --------------------------------------------------------

    original_root = extract_root_domain(
        original_url
    )

    perturbed_root = extract_root_domain(
        perturbed_url
    )

    if perturbation_name != "add_suspicious_subdomain":

        if original_root != perturbed_root:

            raise ValueError(
                f"Root domain changed unexpectedly "
                f"for perturbation '{perturbation_name}': "
                f"{original_root} -> {perturbed_root}"
            )

    else:

        if original_root != perturbed_root:

            raise ValueError(
                "Suspicious-subdomain perturbation "
                "did not preserve root domain."
            )

    return True


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_features(
    model,
    feature_df,
):

    predictions = model.predict(
        feature_df
    )

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    return (
        predictions,
        probabilities,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    unique_classes = np.unique(
        y_true
    )

    if len(unique_classes) < 2:

        roc_auc = np.nan

    else:

        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )

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
            roc_auc,
    }


# ============================================================
# METRIC PRINTING
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print(f"\n{title}:")

    for metric, value in metrics.items():

        if pd.isna(value):

            display_value = "N/A"

        else:

            display_value = f"{value:.4f}"

        print(
            f"{metric.upper():<10}: "
            f"{display_value}"
        )


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
        f"ROBUSTNESS TEST - "
        f"{model_name.upper()}"
    )
    print("=" * 60)

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_features = create_feature_dataframe(
        urls
    )

    baseline_predictions, baseline_probabilities = (
        predict_features(
            model,
            baseline_features,
        )
    )

    baseline_metrics = calculate_metrics(
        labels,
        baseline_predictions,
        baseline_probabilities,
    )

    print_metrics(
        "Baseline performance",
        baseline_metrics,
    )

    print(
        "\nBaseline class distribution:"
    )

    baseline_class_counts = pd.Series(
        labels
    ).value_counts().sort_index()

    for class_value, count in (
        baseline_class_counts.items()
    ):

        print(
            f"Class {class_value}: {count}"
        )

    # ========================================================
    # PERTURBATIONS
    # ========================================================

    all_results = []

    flip_matrix = pd.DataFrame(
        index=range(len(urls))
    )

    for (
        perturbation_name,
        perturbation_function
    ) in PERTURBATIONS.items():

        print("\n")
        print("-" * 60)
        print(
            f"Testing perturbation: "
            f"{perturbation_name}"
        )
        print("-" * 60)

        valid_original_urls = []
        test_urls = []
        test_labels = []
        original_indices = []

        # ----------------------------------------------------
        # Generate perturbations
        # ----------------------------------------------------

        for index, (
            url,
            label
        ) in enumerate(
            zip(urls, labels)
        ):

            perturbed_url = (
                perturbation_function(url)
            )

            if not validate_perturbation(
                url,
                perturbed_url,
                perturbation_name,
            ):
                continue

            valid_original_urls.append(
                url
            )

            test_urls.append(
                perturbed_url
            )

            test_labels.append(
                label
            )

            original_indices.append(
                index
            )

        if not test_urls:

            print(
                "No valid perturbations "
                "generated."
            )

            continue

        test_labels = np.asarray(
            test_labels
        )

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        class_counts = (
            pd.Series(
                test_labels
            )
            .value_counts()
            .sort_index()
        )

        print(
            f"Tests: {len(test_urls)}"
        )

        print(
            "Class distribution:"
        )

        for class_value, count in (
            class_counts.items()
        ):

            print(
                f"  Class {class_value}: "
                f"{count}"
            )

        if len(class_counts) < 2:

            print(
                "NOTE: ROC-AUC is undefined "
                "because this test subset contains "
                "only one class."
            )

        # ----------------------------------------------------
        # Original subset predictions
        # ----------------------------------------------------

        original_features = (
            create_feature_dataframe(
                valid_original_urls
            )
        )

        (
            original_predictions,
            original_probabilities,
        ) = predict_features(
            model,
            original_features,
        )

        # ----------------------------------------------------
        # Original subset metrics
        # ----------------------------------------------------

        original_subset_metrics = (
            calculate_metrics(
                test_labels,
                original_predictions,
                original_probabilities,
            )
        )

        # ----------------------------------------------------
        # Perturbed predictions
        # ----------------------------------------------------

        perturbed_features = (
            create_feature_dataframe(
                test_urls
            )
        )

        (
            perturbed_predictions,
            perturbed_probabilities,
        ) = predict_features(
            model,
            perturbed_features,
        )

        # ----------------------------------------------------
        # Perturbed metrics
        # ----------------------------------------------------

        perturbed_metrics = (
            calculate_metrics(
                test_labels,
                perturbed_predictions,
                perturbed_probabilities,
            )
        )

        # ====================================================
        # PREDICTION CHANGE ANALYSIS
        # ====================================================

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

        prediction_change_rate = (
            prediction_changed.mean()
        )

        robustness_failure_rate = (
            correct_to_incorrect.mean()
        )

        recovery_rate = (
            incorrect_to_correct.mean()
        )

        # ====================================================
        # PROBABILITY ANALYSIS
        # ====================================================

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

        # ====================================================
        # CORRECT DEGRADATION
        # ====================================================

        accuracy_degradation = (
            original_subset_metrics[
                "accuracy"
            ]
            -
            perturbed_metrics[
                "accuracy"
            ]
        )

        f1_degradation = (
            original_subset_metrics[
                "f1"
            ]
            -
            perturbed_metrics[
                "f1"
            ]
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

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

        print_metrics(
            "Original subset performance",
            original_subset_metrics,
        )

        print_metrics(
            "Perturbed performance",
            perturbed_metrics,
        )

        # ====================================================
        # STORE FLIP INFORMATION
        # ====================================================

        for local_index, original_index in enumerate(
            original_indices
        ):

            flip_matrix.loc[
                original_index,
                perturbation_name
            ] = int(
                prediction_changed[
                    local_index
                ]
            )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        all_results.append({

            "perturbation":
                perturbation_name,

            "tests":
                len(test_urls),

            "class_0":
                int(
                    class_counts.get(
                        0,
                        0
                    )
                ),

            "class_1":
                int(
                    class_counts.get(
                        1,
                        0
                    )
                ),

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

            "original_accuracy":
                original_subset_metrics[
                    "accuracy"
                ],

            "original_precision":
                original_subset_metrics[
                    "precision"
                ],

            "original_recall":
                original_subset_metrics[
                    "recall"
                ],

            "original_f1":
                original_subset_metrics[
                    "f1"
                ],

            "original_roc_auc":
                original_subset_metrics[
                    "roc_auc"
                ],

            "accuracy":
                perturbed_metrics[
                    "accuracy"
                ],

            "precision":
                perturbed_metrics[
                    "precision"
                ],

            "recall":
                perturbed_metrics[
                    "recall"
                ],

            "f1":
                perturbed_metrics[
                    "f1"
                ],

            "roc_auc":
                perturbed_metrics[
                    "roc_auc"
                ],
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

        return results_df, flip_matrix

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # FLIP OVERLAP DIAGNOSTIC
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        f"{model_name.upper()} - "
        "PERTURBATION FLIP DIAGNOSTIC"
    )
    print("=" * 60)

    if not flip_matrix.empty:

        flip_matrix = (
            flip_matrix
            .fillna(0)
            .astype(int)
        )

        flip_counts = (
            flip_matrix.sum(
                axis=1
            )
        )

        print(
            "URLs changed by multiple "
            "perturbations:"
        )

        for count in sorted(
            flip_counts.unique()
        ):

            number_of_urls = (
                flip_counts == count
            ).sum()

            print(
                f"  {count} perturbations: "
                f"{number_of_urls} URLs"
            )

        print(
            "\nPairwise perturbation "
            "flip overlap:"
        )

        perturbation_names = list(
            flip_matrix.columns
        )

        for i in range(
            len(perturbation_names)
        ):

            for j in range(
                i + 1,
                len(perturbation_names)
            ):

                first = (
                    perturbation_names[i]
                )

                second = (
                    perturbation_names[j]
                )

                both_flipped = (
                    (
                        flip_matrix[first]
                        == 1
                    )
                    &
                    (
                        flip_matrix[second]
                        == 1
                    )
                ).sum()

                print(
                    f"  {first} + {second}: "
                    f"{both_flipped} URLs"
                )

    return (
        results_df,
        flip_matrix,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    logistic_results,
    random_forest_results,
):

    combined_results = []

    if not logistic_results.empty:

        logistic_copy = (
            logistic_results.copy()
        )

        logistic_copy.insert(
            0,
            "model",
            "Logistic Regression",
        )

        combined_results.append(
            logistic_copy
        )

    if not random_forest_results.empty:

        random_forest_copy = (
            random_forest_results.copy()
        )

        random_forest_copy.insert(
            0,
            "model",
            "Random Forest",
        )

        combined_results.append(
            random_forest_copy
        )

    if not combined_results:

        print(
            "\nNo robustness results "
            "to save."
        )

        return

    combined_df = pd.concat(
        combined_results,
        ignore_index=True,
    )

    combined_df.to_csv(
        ROBUSTNESS_RESULTS_FILE,
        index=False,
    )

    print("\n")
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(
        f"File: "
        f"{ROBUSTNESS_RESULTS_FILE}"
    )

    print(
        f"Rows: {len(combined_df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "QR Phishing Detection - "
        "Robustness Testing"
    )
    print("=" * 60)

    # ========================================================
    # LOAD DATA
    # ========================================================

    feature_df = load_feature_dataset()

    url_df = load_url_dataset()

    verify_alignment(
        feature_df,
        url_df,
    )

    # ========================================================
    # CREATE UNSEEN-ROOT-DOMAIN HOLDOUT
    # ========================================================

    (
        train_indices,
        test_indices,
        groups,
    ) = create_robustness_holdout(
        url_df
    )

    # ========================================================
    # TRAINING DATA
    # ========================================================

    X_train = (
        feature_df[
            FEATURE_NAMES
        ]
        .iloc[
            train_indices
        ]
        .copy()
    )

    y_train = (
        feature_df[
            "label"
        ]
        .iloc[
            train_indices
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Feature schema validation
    # --------------------------------------------------------

    if list(
        X_train.columns
    ) != FEATURE_NAMES:

        raise ValueError(
            "Training feature schema does not "
            "match FEATURE_NAMES."
        )

    print(
        "\nPASS: Training feature schema "
        "matches FEATURE_NAMES."
    )

    # ========================================================
    # ROBUSTNESS TEST DATA
    # ========================================================

    robustness_url_df = (
        url_df
        .iloc[
            test_indices
        ]
        .copy()
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    sample_size = min(
        ROBUSTNESS_SAMPLE_SIZE,
        len(robustness_url_df),
    )

    robustness_sample = (
        robustness_url_df
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nRobustness URLs sampled: "
        f"{len(robustness_sample)}"
    )

    # --------------------------------------------------------
    # Sample class validation
    # --------------------------------------------------------

    sample_classes = (
        robustness_sample[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    if len(sample_classes) < 2:

        raise ValueError(
            "Robustness sample contains only "
            "one class. Both classes are required "
            "for the primary robustness evaluation."
        )

    print(
        "PASS: Robustness sample contains "
        "both classes."
    )

    urls = (
        robustness_sample[
            "URL"
        ]
        .tolist()
    )

    labels = (
        robustness_sample[
            "label"
        ]
        .to_numpy()
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

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

    # ========================================================
    # RANDOM FOREST
    # ========================================================

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

    # ========================================================
    # LOGISTIC ROBUSTNESS
    # ========================================================

    (
        logistic_results,
        logistic_flip_matrix,
    ) = evaluate_robustness(
        "Logistic Regression",
        logistic_model,
        urls,
        labels,
    )

    # ========================================================
    # RANDOM FOREST ROBUSTNESS
    # ========================================================

    (
        random_forest_results,
        random_forest_flip_matrix,
    ) = evaluate_robustness(
        "Random Forest",
        random_forest_model,
        urls,
        labels,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        logistic_results,
        random_forest_results,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "ROBUSTNESS TEST COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()