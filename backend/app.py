from fastapi import FastAPI
from pydantic import BaseModel
from services.predictor import predict_url

app = FastAPI(
    title="QR XAI Phishing Detection API",
    description="Backend API for QR phishing detection with risk explanation",
    version="0.1.0"
)

class URLRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "QR XAI Phishing Detection API is running"}

@app.post("/predict")
def predict(data: URLRequest):
    return predict_url(data.url)