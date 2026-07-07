from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
import sys
import uvicorn
import json

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.predict import load_model, predict_fraud, predict_batch

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions",
    version="1.0.0"
)


def dump_transaction(transaction_model):
    """Return a plain dictionary for a Pydantic model."""
    if hasattr(transaction_model, "model_dump"):
        return transaction_model.model_dump()
    return transaction_model.dict()

# Load the model at startup
model = None


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    if model is None:
        print("Warning: Model could not be loaded. API will not function correctly.")


# Define request and response models
class TransactionRequest(BaseModel):
    trans_date_trans_time: str = Field(..., description="Transaction timestamp")
    cc_num: str = Field(..., description="Credit card number")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., description="Transaction amount")
    first: str = Field(..., description="Cardholder first name")
    last: str = Field(..., description="Cardholder last name")
    gender: str = Field(..., description="Cardholder gender")
    street: str = Field(..., description="Cardholder street address")
    city: str = Field(..., description="Cardholder city")
    state: str = Field(..., description="Cardholder state")
    zip: str = Field(..., description="Cardholder ZIP code")
    lat: float = Field(..., description="Cardholder latitude")
    long: float = Field(..., description="Cardholder longitude")
    city_pop: int = Field(..., description="City population")
    job: str = Field(..., description="Cardholder job")
    dob: str = Field(..., description="Cardholder date of birth")
    trans_num: str = Field(..., description="Transaction number")
    unix_time: int = Field(..., description="Unix timestamp")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., description="Probability of fraud")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., description="List of transactions")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


@app.get("/")
async def root():
    return {"message": "Welcome to the Fraud Detection API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert Pydantic model to dictionary
    transaction_dict = dump_transaction(transaction)

    # Make prediction
    result = predict_fraud(model, transaction_dict)

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_multiple(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert Pydantic models to dictionaries
    transactions_dict = [dump_transaction(transaction) for transaction in request.transactions]

    # Make predictions
    results = predict_batch(model, transactions_dict)

    return {"predictions": results}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
async def model_info():
    try:
        with open(config.MODEL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metadata not found")


def main():
    """Run the API server"""
    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )


if __name__ == "__main__":
    main()
