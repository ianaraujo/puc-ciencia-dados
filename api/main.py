import pickle
import pandas as pd

from fastapi import FastAPI, HTTPException, status

from src.preprocess import FeatureTransform, preprocess_data


app = FastAPI()

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(request: dict):
    data = pd.DataFrame([request])
    data_transformed = FeatureTransform.create_features(data)
    
    X = preprocess_data(data_transformed)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    if y_pred[0] == 0:
        return {"pred": y_proba[0].tolist()}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model indicates this transaction is likely to be fraudulent!"
        )