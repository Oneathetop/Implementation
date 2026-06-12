from fastapi import FastAPI  
from pydantic import BaseModel
from services.predictor import predict_url

app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict(data: URLRequest):

    result = predict_url(data.url)

    return {
        "url": data.url,
        **result
    }