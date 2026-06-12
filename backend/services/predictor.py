def predict_url(url):

    url = url.lower()

    reasons = []
    risk_score = 0.1

    if not url.startswith("https"):
        risk_score += 0.3
        reasons.append("URL does not use HTTPS")

    if "login" in url or "verify" in url or "account" in url:
        risk_score += 0.3
        reasons.append("Suspicious keyword detected")

    if len(url) > 60:
        risk_score += 0.2
        reasons.append("URL is unusually long")

    prediction = "Malicious" if risk_score >= 0.5 else "Safe"

    return {
        "prediction": prediction,
        "risk_score": round(risk_score, 2),
        "explanation": reasons if reasons else [
            "No major suspicious patterns detected"
        ]
    }