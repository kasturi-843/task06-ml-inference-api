from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Titanic ML Prediction API",
    description="Real-Time Machine Learning Inference API",
    version="1.0.0"
)

# Load champion model
model = joblib.load("model/task4_best_model.joblib")


class PredictionInput(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str
    class_: str
    who: str
    alone: str


@app.get("/")
def home():
    return {
        "message": "Titanic ML Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{
        "pclass": data.pclass,
        "sex": data.sex,
        "age": data.age,
        "sibsp": data.sibsp,
        "parch": data.parch,
        "fare": data.fare,
        "embarked": data.embarked,
        "class": data.class_,
        "who": data.who,
        "alone": data.alone
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]
    probability = float(max(probabilities))

    return {
        "prediction": int(prediction),
        "probability": probability
    }