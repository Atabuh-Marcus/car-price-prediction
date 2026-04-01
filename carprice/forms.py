from django import forms

class CarPriceForm(forms.Form):
    levy = forms.FloatField(label='Levy', required=True)
    prod_year = forms.IntegerField(label='Production Year', required=True)
    engine_volume = forms.FloatField(label='Engine Volume', required=True)
    mileage = forms.FloatField(label='Mileage', required=True)
    cylinders = forms.IntegerField(label='Cylinders', required=True)
    airbags = forms.IntegerField(label='Airbags', required=True)
    leather_interior = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Leather Interior', required=True)
    manufacturer = forms.CharField(label='Manufacturer', max_length=100, required=True)
    model = forms.CharField(label='Model', max_length=100, required=True)
    category = forms.CharField(label='Category', max_length=100, required=True)
    fuel_type = forms.CharField(label='Fuel Type', max_length=100, required=True)
    gear_box = forms.CharField(label='Gear Box Type', max_length=100, required=True)
    drive_wheels = forms.CharField(label='Drive Wheels', max_length=100, required=True)
    wheel = forms.CharField(label='Wheel', max_length=100, required=True)
    color = forms.CharField(label='Color', max_length=100, required=True)