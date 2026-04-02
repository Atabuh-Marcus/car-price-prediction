import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

lr_model    = joblib.load('car_price_model.pkl')
rf_model    = joblib.load('car_price_rf_model.pkl')
stack_model = joblib.load('car_price_stack_model.pkl')
scaler      = joblib.load('scaler.pkl')
scaler2     = joblib.load('scaler2.pkl')
le_mfr      = joblib.load('manufacturer_encoder.pkl')
le_mdl      = joblib.load('model_encoder.pkl')
le_cat      = joblib.load('category_encoder.pkl')
le_fuel     = joblib.load('fuel_encoder.pkl')
le_gear     = joblib.load('gear_encoder.pkl')
le_drive    = joblib.load('drive_encoder.pkl')
le_wheel    = joblib.load('wheel_encoder.pkl')
le_color    = joblib.load('color_encoder.pkl')

df = pd.read_csv('car_price_prediction.csv')
df.replace('-', np.nan, inplace=True)
df.dropna(inplace=True)
df['Levy']          = pd.to_numeric(df['Levy'])
df['Mileage']       = df['Mileage'].str.replace(' km','').str.replace(',','').astype(float)
df['Engine volume'] = df['Engine volume'].str.extract(r'([\d.]+)').astype(float)
df['Prod. year']    = pd.to_numeric(df['Prod. year'])
df['Cylinders']     = pd.to_numeric(df['Cylinders'])
df['Airbags']       = pd.to_numeric(df['Airbags'])
df['Price']         = pd.to_numeric(df['Price'])
df['Manufacturer_enc'] = le_mfr.transform(df['Manufacturer'])
df['Model_enc']     = le_mdl.transform(df['Model'])
df['Category_enc']  = le_cat.transform(df['Category'])
df['Fuel_enc']      = le_fuel.transform(df['Fuel type'])
df['Gear_enc']      = le_gear.transform(df['Gear box type'])
df['Drive_enc']     = le_drive.transform(df['Drive wheels'])
df['Wheel_enc']     = le_wheel.transform(df['Wheel'])
df['Color_enc']     = le_color.transform(df['Color'])
df['Leather interior'] = df['Leather interior'].map({'Yes':1,'No':0})

features = ['Levy','Prod. year','Engine volume','Mileage','Cylinders','Airbags',
            'Leather interior','Manufacturer_enc','Model_enc','Category_enc',
            'Fuel_enc','Gear_enc','Drive_enc','Wheel_enc','Color_enc']
X = df[features]
y = df['Price']
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, X_test3, _, y_test3 = train_test_split(X, y_log, test_size=0.2, random_state=42)

y_pred_lr    = lr_model.predict(scaler.transform(X_test))
y_pred_rf    = np.expm1(rf_model.predict(scaler2.transform(X_test)))
y_pred_stack = np.expm1(stack_model.predict(X_test3))
y_test3_orig = np.expm1(y_test3)

print("=" * 62)
print(f"{'Model':<30} {'MAE':>8} {'RMSE':>10} {'R2':>8}")
print("=" * 62)
for name, yt, yp in [
    ('Linear Regression',       y_test,      y_pred_lr),
    ('Random Forest',           y_test,      y_pred_rf),
    ('Stacking (LR+RF->Ridge)', y_test3_orig, y_pred_stack),
]:
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    print(f"{name:<30} {mae:>8.2f} {rmse:>10.2f} {r2:>8.4f}")
print("=" * 62)

# 5 real test rows to manually verify
print("\n--- 5 Sample Rows to Test ---")
print(f"{'Actual':>10}  {'LR':>10}  {'RF':>10}  {'Stack':>10}")
for a, lr, rf, st in list(zip(y_test, y_pred_lr, y_pred_rf, y_pred_stack))[:5]:
    print(f"${a:>9,.0f}  ${lr:>9,.0f}  ${rf:>9,.0f}  ${st:>9,.0f}")

# Print 3 example inputs from the test set with raw feature values
print("\n--- 3 Raw Input Examples from Test Set ---")
raw = df.loc[X_test.index[:3], ['Manufacturer','Model','Category','Fuel type',
                                  'Gear box type','Drive wheels','Wheel','Color',
                                  'Levy','Prod. year','Engine volume','Mileage',
                                  'Cylinders','Airbags','Leather interior','Price']]
print(raw.to_string())
