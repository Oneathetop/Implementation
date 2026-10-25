from pathlib import Path

import joblib
import pandas as pd

from ml.config import FEATURE_NAMES
from ml.feature_extractor import extract_features

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

    if predicted_label == 0:
        prediction = "Malicious"
    else:
        prediction = "Safe"

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

    explanation = explain_prediction(
        model=random_forest_model,
        input_data=input_data,
        predicted_label=predicted_label,
        phishing_probability=phishing_probability,
    )

    return {
        "url": original_url,
        "prediction": prediction,
        "risk_score": round(phishing_probability, 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "explanation": explanation,
        "recommendation": recommendation,
    }