def predict_url(url: str):
    original_url = url
    url = url.lower().strip()

    reasons = []
    risk_score = 0.1

    if not url.startswith("https://"):
        risk_score += 0.30
        reasons.append("URL does not use HTTPS")

    suspicious_keywords = ["login", "verify", "account", "secure", "update", "bank", "password"]
    found_keywords = [word for word in suspicious_keywords if word in url]

    if found_keywords:
        risk_score += 0.25
        reasons.append(
            f"Suspicious keyword detected: {', '.join(found_keywords)}"
        )

    if len(url) > 60:
        risk_score += 0.20
        reasons.append("URL is unusually long")

    special_chars = ["@", "-", "=", "%", "?"]
    special_char_count = sum(url.count(char) for char in special_chars)

    if special_char_count >= 4:
        risk_score += 0.15
        reasons.append("URL contains many special characters")

    risk_score = min(risk_score, 0.99)

    if risk_score >= 0.70:
        prediction = "Malicious"
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = "Do not open this link. Verify the source before continuing."

    elif risk_score >= 0.40:
        prediction = "Suspicious"
        risk_level = "Medium Risk"
        risk_color = "amber"
        recommendation = "Be cautious. Check the sender and destination before opening."

    else:
        prediction = "Safe"
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = "No major suspicious patterns were detected, but always verify unknown QR codes."

    return {
        "url": original_url,
        "prediction": prediction,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "explanation": reasons if reasons else ["No major suspicious patterns detected"],
        "recommendation": recommendation
    }