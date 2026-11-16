from pathlib import Path

import joblib
import pandas as pd

from ml.config import FEATURE_NAMES
from ml.feature_extractor import extract_features

from services.domain_policy import (
    find_brand_lookalike,
    get_root_domain,
    is_trusted_root_domain,
)
from services.explainer import explain_prediction

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "random_forest_model.joblib"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model not found at: {MODEL_PATH}"
    )


random_forest_model = joblib.load(MODEL_PATH)


model_features = list(
    getattr(random_forest_model, "feature_names_in_", [])
)

if model_features and model_features != FEATURE_NAMES:
    raise RuntimeError(
        "The model feature order does not match FEATURE_NAMES."
    )


def predict_url(url: str) -> dict:
    original_url = url.strip()

    if not original_url:
        raise ValueError("URL cannot be empty.")

    extracted_features = extract_features(original_url)

    feature_row = {
        feature_name: extracted_features[feature_name]
        for feature_name in FEATURE_NAMES
    }

    input_data = pd.DataFrame(
        [feature_row],
        columns=FEATURE_NAMES,
    )

    predicted_label = int(
        random_forest_model.predict(input_data)[0]
    )

    probabilities = random_forest_model.predict_proba(input_data)[0]
    phishing_class_index = list(
        random_forest_model.classes_
    ).index(0)

    phishing_probability = float(
        probabilities[phishing_class_index]
    )

    model_prediction = "Malicious" if predicted_label == 0 else "Safe"
    model_risk_score = phishing_probability
    root_domain = get_root_domain(original_url)
    trusted_domain = is_trusted_root_domain(original_url)
    brand_lookalike = find_brand_lookalike(original_url)

    if trusted_domain and extracted_features["https"] == 1:
        prediction = "Safe"
        phishing_probability = min(phishing_probability, 0.39)
    elif brand_lookalike:
        prediction = "Malicious"
        phishing_probability = max(phishing_probability, 0.90)
    else:
        prediction = model_prediction

    if phishing_probability >= 0.70:
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = (
            "Do not open this link. "
            "Verify the source before continuing."
        )
    elif phishing_probability >= 0.40:
        risk_level = "Medium Risk"
        risk_color = "amber"
        recommendation = (
            "Be cautious. Check the sender and destination "
            "before opening."
        )
    else:
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = (
            "No major suspicious patterns were detected, "
            "but always verify unknown QR codes."
        )

    if trusted_domain and extracted_features["https"] == 1:
        explanation = [
            (
                f"The URL uses the recognized root domain "
                f"'{root_domain}'. The model's structural warning "
                "was reduced, but verify the specific page and sender."
            ),
            (
                "The domain match does not verify the page content, "
                "redirects, or the message that contained this link."
            ),
        ]
    else:
        explanation = explain_prediction(
            model=random_forest_model,
            input_data=input_data,
            predicted_label=predicted_label,
            phishing_probability=model_risk_score,
        )

    if brand_lookalike:
        explanation.insert(
            0,
            (
                f"The root domain appears to imitate the brand "
                f"'{brand_lookalike}'."
            ),
        )

    return {
        "url": original_url,
        "prediction": prediction,
        "risk_score": round(phishing_probability, 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "root_domain": root_domain,
        "domain_status": (
            "trusted"
            if trusted_domain and extracted_features["https"] == 1
            else "untrusted"
        ),
        "explanation": explanation,
        "recommendation": recommendation,
    }
