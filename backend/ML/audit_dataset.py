import pandas as pd

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split

from ml.config import (
    CLEAN_DATASET,
    TEST_SIZE,
    RANDOM_STATE,
)


def main():

    print("=" * 60)
    print("QR Phishing Detection - Dataset Leakage Audit")
    print("=" * 60)

    df = load_dataset()

    check_exact_duplicates(df)

    check_label_conflicts(df)

    analyze_domains(df)

    analyze_domain_overlap(df)


def load_dataset():

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(CLEAN_DATASET)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def check_exact_duplicates(df):

    print("\n" + "-" * 60)
    print("1. EXACT DUPLICATE URL CHECK")
    print("-" * 60)

    duplicate_count = df["URL"].duplicated().sum()

    print(f"Duplicate URLs: {duplicate_count}")

    if duplicate_count == 0:
        print("PASS: No exact duplicate URLs found.")
    else:
        print("WARNING: Duplicate URLs found.")


def check_label_conflicts(df):

    print("\n" + "-" * 60)
    print("2. DUPLICATE LABEL CONSISTENCY CHECK")
    print("-" * 60)

    label_counts = (
        df.groupby("URL")["label"]
        .nunique()
    )

    conflicting_urls = (label_counts > 1).sum()

    print(f"URLs with conflicting labels: {conflicting_urls}")

    if conflicting_urls == 0:
        print("PASS: No URL has conflicting labels.")
    else:
        print("WARNING: Conflicting labels detected.")


def extract_domain(url):

    try:

        parsed = urlparse(url)

        hostname = parsed.hostname

        if hostname:
            return hostname.lower()

        return ""

    except Exception:

        return ""


def analyze_domains(df):

    print("\n" + "-" * 60)
    print("3. DOMAIN ANALYSIS")
    print("-" * 60)

    domains = df["URL"].apply(extract_domain)

    unique_domains = domains.nunique()

    empty_domains = (domains == "").sum()

    print(f"Unique domains: {unique_domains}")
    print(f"URLs with empty domain: {empty_domains}")

    domain_counts = domains.value_counts()

    print("\nTop 10 most frequent domains:")

    print(domain_counts.head(10))


def analyze_domain_overlap(df):

    print("\n" + "-" * 60)
    print("4. TRAIN / TEST DOMAIN OVERLAP")
    print("-" * 60)

    X = df["URL"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,
    )

    train_domains = set(
        X_train.apply(extract_domain)
    )

    test_domains = set(
        X_test.apply(extract_domain)
    )

    overlapping_domains = train_domains.intersection(
        test_domains
    )

    unseen_test_domains = test_domains - train_domains

    print(f"Training domains: {len(train_domains)}")
    print(f"Testing domains: {len(test_domains)}")
    print(f"Overlapping domains: {len(overlapping_domains)}")
    print(f"Unseen test domains: {len(unseen_test_domains)}")

    test_domain_series = X_test.apply(extract_domain)

    seen_domain_test_urls = test_domain_series.isin(
        train_domains
    ).sum()

    unseen_domain_test_urls = (
        ~test_domain_series.isin(train_domains)
    ).sum()

    print(
        f"\nTest URLs from domains seen during training: "
        f"{seen_domain_test_urls}"
    )

    print(
        f"Test URLs from completely unseen domains: "
        f"{unseen_domain_test_urls}"
    )

    overlap_percentage = (
        seen_domain_test_urls / len(X_test)
    ) * 100

    unseen_percentage = (
        unseen_domain_test_urls / len(X_test)
    ) * 100

    print(
        f"\nPercentage of test URLs from seen domains: "
        f"{overlap_percentage:.2f}%"
    )

    print(
        f"Percentage of test URLs from unseen domains: "
        f"{unseen_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()