from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from ml.config import FEATURE_NAMES


FEATURE_DESCRIPTIONS = {
    "url_length": "The URL is unusually long.",
    "domain_length": "The domain name is unusually long.",
    "path_length": "The URL path is unusually long.",
    "slash_count": "The URL contains an unusual number of path separators.",
    "double_slash_in_path": (
        "The URL contains a double slash inside its path."
    ),
    "suspicious_file_extension": (
        "The URL points to a potentially dangerous file type."
    ),
    "subdomain_count": "The URL contains multiple subdomains.",
    "query_parameter_count": (
        "The URL contains multiple query parameters."
    ),
    "https": "The HTTPS feature contributed to the classification.",
    "ip_address": (
        "The IP-address feature contributed to the classification."
    ),
    "dot_count": "The URL contains an unusual number of dots.",
    "digit_count": "The URL contains an unusual number of digits.",
    "max_consecutive_digits": (
        "The URL contains a long consecutive sequence of digits."
    ),
    "hyphen_count": "The URL contains multiple hyphens.",
    "special_character_count": (
        "The URL contains many special characters."
    ),
    "at_symbol_presence": (
        "The URL contains an @ symbol, which can hide the true destination."
    ),
    "suspicious_character_ratio": (
        "A relatively large portion of the URL consists of suspicious "
        "characters."
    ),
    "url_encoding_count": (
        "The URL contains encoded characters that may obscure its destination."
    ),
    "entropy": (
        "The URL has a high degree of randomness in its character sequence."
    ),
    "consecutive_special_character_count": (
        "The URL contains consecutive special characters."
    ),
    "keyword_count": (
        "The URL contains phishing-related keywords such as login, verify, "
        "account, or password."
    ),
    "brand_keyword_count": (
        "The URL contains a recognizable brand name."
    ),
    "url_shortener": (
        "The URL-shortener feature contributed to the classification."
    ),
    "port_number": (
        "The URL explicitly specifies a network port."
    ),
    "suspicious_tld": (
        "The top-level-domain feature contributed to the classification."
    ),
    "fragment_present": (
        "The URL contains a fragment that may be used to obscure content."
    ),
}


def _get_class_shap_values(
    shap_values: Any,
    class_index: int,
) -> np.ndarray:
    """
    Normalize SHAP output across supported SHAP return formats.
    """

    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index][0])

    values = np.asarray(shap_values)

    if values.ndim == 3:
        return values[0, :, class_index]

    if values.ndim == 2:
        return values[0]

    raise RuntimeError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


def _format_reason(
    feature_name: str,
    feature_value: Any,
    contribution: float,
) -> str:
    """
    Convert a model feature contribution into an accurate explanation.
    """

    if feature_name == "https":
        if feature_value == 0:
            description = "The URL does not use HTTPS."
        else:
            description = (
                "The HTTPS feature contributed to the model's "
                "phishing classification."
            )

    elif feature_name == "ip_address":
        if feature_value == 1:
            description = (
                "The URL uses an IP address instead of a normal domain name."
            )
        else:
            description = (
                "The IP-address feature contributed to the model's "
                "phishing classification."
            )

    elif feature_name == "url_shortener":
        if feature_value == 1:
            description = "The URL uses a URL-shortening service."
        else:
            description = (
                "The URL-shortener feature contributed to the model's "
                "phishing classification."
            )

    elif feature_name == "suspicious_tld":
        if feature_value == 1:
            description = (
                "The URL uses a top-level domain associated with "
                "suspicious URLs."
            )
        else:
            description = (
                "The top-level-domain feature contributed to the model's "
                "phishing classification."
            )

    else:
        description = FEATURE_DESCRIPTIONS.get(
            feature_name,
            f"The feature '{feature_name}' contributed to the prediction.",
        )

    return (
        f"{description} "
        f"Observed value: {feature_value}. "
        f"Model contribution toward phishing: "
        f"{contribution:.4f}."
    )


def explain_prediction(
    model: Any,
    input_data: pd.DataFrame,
    predicted_label: int,
    phishing_probability: float,
    max_reasons: int = 5,
) -> list[str]:
    """
    Generate local explanations for one URL prediction.

    Label 0 is treated as phishing, matching the project's
    Random Forest evaluation code.
    """

    if predicted_label != 0:
        return [
            "The model did not identify strong phishing indicators "
            "for this URL."
        ]

    if list(input_data.columns) != FEATURE_NAMES:
        raise ValueError(
            "The input feature order does not match FEATURE_NAMES."
        )

    shap = importlib.import_module("shap")

    class_index = list(model.classes_).index(0)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data,
        check_additivity=False,
    )

    class_values = _get_class_shap_values(
        shap_values,
        class_index,
    )

    if len(class_values) != len(FEATURE_NAMES):
        raise RuntimeError(
            "The number of SHAP values does not match the feature count."
        )

    contributions = sorted(
        zip(FEATURE_NAMES, class_values),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    positive_contributions = [
        item
        for item in contributions
        if float(item[1]) > 0
    ]

    explanations = [
        (
            f"The model estimated a "
            f"{phishing_probability:.2%} probability that this URL "
            "is phishing."
        )
    ]

    for feature_name, contribution in positive_contributions[:max_reasons]:
        feature_value = input_data.iloc[0][feature_name]

        explanations.append(
            _format_reason(
                feature_name=feature_name,
                feature_value=feature_value,
                contribution=float(contribution),
            )
        )

    if len(explanations) == 1:
        explanations.append(
            "The Random Forest classified the URL as phishing, "
            "but no individual feature had a strong positive local "
            "contribution."
        )

    return explanations