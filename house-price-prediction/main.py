from src.data_preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate
from src.predict import predict_price

X, y = preprocess_data("data/train.csv")
model, X_test, y_test = train_model(X, y)
evaluate(model, X_test, y_test)

sample_house = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "TotalBsmtSF": 900,
    "BedroomAbvGr": 3,
    "FullBath": 2
}

price = predict_price(sample_house)
print("Predicted House Price:", price)
