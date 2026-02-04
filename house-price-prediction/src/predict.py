import pandas as pd
import joblib

def predict_price(input_data):
    model = joblib.load("outputs/model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    return model.predict(input_scaled)[0]
