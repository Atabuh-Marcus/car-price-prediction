from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Car Price Prediction API")

# Load the model and preprocessing objects
try:
    lr_model = joblib.load('../car_price_model.pkl')
    scaler = joblib.load('../scaler.pkl')
    le_manufacturer = joblib.load('../manufacturer_encoder.pkl')
    le_model = joblib.load('../model_encoder.pkl')
    le_category = joblib.load('../category_encoder.pkl')
    le_fuel = joblib.load('../fuel_encoder.pkl')
    le_gear = joblib.load('../gear_encoder.pkl')
    le_drive = joblib.load('../drive_encoder.pkl')
    le_wheel = joblib.load('../wheel_encoder.pkl')
    le_color = joblib.load('../color_encoder.pkl')
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")

# Define the input data model
class CarFeatures(BaseModel):
    levy: float
    prod_year: int
    engine_volume: float
    mileage: float
    cylinders: int
    airbags: int
    leather_interior: int  # 1 for Yes, 0 for No
    manufacturer: str
    model: str
    category: str
    fuel_type: str
    gear_box: str
    drive_wheels: str
    wheel: str
    color: str

@app.post("/predict")
def predict_price(car: CarFeatures):
    try:
        # Encode categorical variables
        manufacturer_enc = le_manufacturer.transform([car.manufacturer])[0]
        model_enc = le_model.transform([car.model])[0]
        category_enc = le_category.transform([car.category])[0]
        fuel_enc = le_fuel.transform([car.fuel_type])[0]
        gear_enc = le_gear.transform([car.gear_box])[0]
        drive_enc = le_drive.transform([car.drive_wheels])[0]
        wheel_enc = le_wheel.transform([car.wheel])[0]
        color_enc = le_color.transform([car.color])[0]

        # Create feature array
        features = np.array([[car.levy, car.prod_year, car.engine_volume, car.mileage, car.cylinders, car.airbags, car.leather_interior, manufacturer_enc, model_enc, category_enc, fuel_enc, gear_enc, drive_enc, wheel_enc, color_enc]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = lr_model.predict(features_scaled)[0]

        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API"} 