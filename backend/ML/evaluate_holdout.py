import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from ml.config import FEATURE_NAMES, HOLDOUT_DATASET
from ml.feature_extractor import extract_features
from services.predictor import random_forest_model


def build_features(urls):
    rows = []

    for url in urls:
        extracted = extract_features(url)
        rows.append(
            {
                feature_name: extracted[feature_name]
                for feature_name in FEATURE_NAMES
            }
        )

    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def main():
    holdout = pd.read_csv(HOLDOUT_DATASET)
    required_columns = {"URL", "label"}

    if not required_columns.issubset(holdout.columns):
        raise ValueError("Holdout dataset must contain URL and label columns.")

    features = build_features(holdout["URL"].astype(str))
    predictions = random_forest_model.predict(features)

    print("Curated holdout results")
    print(f"Samples: {len(holdout)}")
    print(f"Accuracy: {accuracy_score(holdout['label'], predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(holdout['label'], predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall: "
        f"{recall_score(holdout['label'], predictions, zero_division=0):.4f}"
    )
    print("Confusion matrix:")
    print(confusion_matrix(holdout["label"], predictions))
    print("Classification report:")
    print(classification_report(holdout["label"], predictions, zero_division=0))

    results = holdout[["URL", "label"]].copy()
    results["prediction"] = predictions
    print("Per-URL results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
