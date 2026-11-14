from fastapi import FastAPI
from pydantic import BaseModel
from services.predictor import predict_url
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="QR XAI Phishing Detection API",
    description="Backend API for QR phishing detection with risk explanation",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "QR XAI Phishing Detection API is running"}

@app.post("/predict")
def predict(data: URLRequest):
    return predict_url(data.url)