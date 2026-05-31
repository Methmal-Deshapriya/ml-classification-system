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

@app.get("/options")
def get_options():
    return {
        "area_type_options": AREA_TYPE_OPTIONS,
        "location_options": LOCATION_OPTIONS,
    }

def create_model_input_row(data: HousePricePredictionInput):
    # Create an empty row with the same number of features used during training
    input_row = np.zeros(len(feature_columns))

    # These are the numerical values we can directly place into the row
    numerical_values = {
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk,
    }

    # Fill numerical feature values into the correct column positions
    for feature_name, value in numerical_values.items():
        feature_index = feature_columns.index(feature_name)
        input_row[feature_index] = value

    # Validate and encode area type
    if data.area_type not in AREA_TYPE_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid area_type. Valid options are: {AREA_TYPE_OPTIONS}"
        )

    area_type_index = feature_columns.index(data.area_type)
    input_row[area_type_index] = 1

    # Validate and encode location
    cleaned_location = data.location.strip()

    if cleaned_location not in LOCATION_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid location. Please select a location from the available options."
        )

    location_index = feature_columns.index(cleaned_location)
    input_row[location_index] = 1

    return input_row

@app.post("/predict")
def predict_house_price(data: HousePricePredictionInput):
    # Convert user-friendly input into model-ready numerical input
    input_row = create_model_input_row(data)

    # Send the prepared input row to the trained model
    predicted_price = model.predict([input_row])[0]

    # Return the predicted price as JSON
    return {
        "predicted_price_lakhs": round(float(predicted_price), 2),
        "input_received": {
            "area_type": data.area_type,
            "location": data.location,
            "total_sqft": data.total_sqft,
            "bath": data.bath,
            "bhk": data.bhk,
            "balcony": data.balcony,
        }
    }