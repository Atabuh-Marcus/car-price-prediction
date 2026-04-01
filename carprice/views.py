from django.shortcuts import render
import requests
from .forms import CarPriceForm
from django.conf import settings

def predict_car_price(request):
    prediction = None
    if request.method == 'POST':
        form = CarPriceForm(request.POST)
        if form.is_valid():
            # Prepare data for FastAPI
            data = {
                'levy': form.cleaned_data['levy'],
                'prod_year': form.cleaned_data['prod_year'],
                'engine_volume': form.cleaned_data['engine_volume'],
                'mileage': form.cleaned_data['mileage'],
                'cylinders': form.cleaned_data['cylinders'],
                'airbags': form.cleaned_data['airbags'],
                'leather_interior': int(form.cleaned_data['leather_interior']),
                'manufacturer': form.cleaned_data['manufacturer'],
                'model': form.cleaned_data['model'],
                'category': form.cleaned_data['category'],
                'fuel_type': form.cleaned_data['fuel_type'],
                'gear_box': form.cleaned_data['gear_box'],
                'drive_wheels': form.cleaned_data['drive_wheels'],
                'wheel': form.cleaned_data['wheel'],
                'color': form.cleaned_data['color'],
            }
            # Call FastAPI
            try:
                response = requests.post(settings.FASTAPI_URL + '/predict', json=data)
                if response.status_code == 200:
                    prediction = response.json()['predicted_price']
                else:
                    prediction = "Error: " + response.text
            except requests.exceptions.RequestException as e:
                prediction = f"Error connecting to prediction service: {str(e)}"
    else:
        form = CarPriceForm()
    return render(request, 'carprice/predict.html', {'form': form, 'prediction': prediction})
