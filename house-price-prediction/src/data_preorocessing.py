import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "BedroomAbvGr",
    "FullBath"
]

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df["SalePrice"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "outputs/scaler.pkl")

    return X_scaled, y
