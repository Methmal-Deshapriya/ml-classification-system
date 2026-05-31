from fastapi import FastAPI, HTTPException #to create the fast api app and to send error responses if something goes wrong
from fastapi.middleware.cors import CORSMiddleware #import CORS config to imply CORS security
from pydantic import BaseModel #to create the expected request body
from pathlib import Path #works well with file paths
import pickle #to load the trained model
import numpy as np #used to create the input row for the model


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app = FastAPI(
    title="House Price Prediction API",
    description="A simple ML regression API for predicting house prices.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#getting the model's path
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bengaluru_house_price_linear_regression_model.pickle"
print(f"Model path: {MODEL_PATH}")



try:
    with open(MODEL_PATH, "rb") as file:
        model_package = pickle.load(file)

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

except FileNotFoundError:
    raise RuntimeError(f"model not found at {MODEL_PATH}")

except KeyError:
    raise RuntimeError("Model file does not contain expected keys: 'model' and 'feature_columns'")


NUMERICAL_FEATURES = ["total_sqft", "bath", "balcony", "bhk"]

AREA_TYPE_OPTIONS = [
    "Built-up  Area",
    "Carpet  Area",
    "Plot  Area",
    "Super built-up  Area",
]

LOCATION_OPTIONS = [
    column for column in feature_columns
    if column not in NUMERICAL_FEATURES and column not in AREA_TYPE_OPTIONS
]

class HousePricePredictionInput(BaseModel):
    area_type: str
    location: str
    total_sqft: float
    bath: float
    bhk: int
    balcony: float


@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {"model_type": type(model).__name__, "number_of_features": len(feature_columns), "sample_feature_columns": feature_columns[:10]}

