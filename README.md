# Car Price Prediction Web App

This project integrates a car price prediction model into a web interface using Django for the frontend and FastAPI for the prediction API.

## Project Structure

- `carpriceweb/`: Django project
- `carprice/`: Django app for the prediction interface
- `fastapi_app/`: FastAPI service for predictions
- `car_price_prediction.csv`: Dataset
- `car_price_model.pkl`, `scaler.pkl`, and encoder `.pkl` files: Trained model and preprocessing objects

## Setup

1. Install dependencies (already done):
   - Django
   - FastAPI
   - Uvicorn
   - Scikit-learn
   - Pandas
   - NumPy
   - Joblib

2. Run the FastAPI server:
   ```
   cd fastapi_app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Run the Django server:
   ```
   python manage.py runserver
   ```

4. Open http://localhost:8000 in your browser for the API docs.
5. Open http://localhost:8000 for the Django web interface.

## Usage

- Use the Django form to input car features.
- The app sends the data to the FastAPI service for prediction.
- The predicted price is displayed.

## API Endpoint

- POST /predict: Accepts car features JSON, returns predicted price.