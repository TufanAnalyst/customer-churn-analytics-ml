from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/best_churn_pipeline.joblib")

class UserInput(BaseModel):
    credit_score: int
    country: str
    gender: str
    age_group: str
    tenure_group: str
    balance: float
    products_number: int
    credit_card: str
    active_member: str
    estimated_salary: float
    has_zero_balance: str
    inactive_with_one_product: str

@app.post("/predict")
def predict(data: UserInput):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    if prediction == 1:
        return {
            "result": "Churn",
            "churn_probability": round(float(proba[1]), 4),
            "retention_probability": round(float(proba[0]), 4)
        }
    else:
        return {
            "result": "Not Churn",
            "churn_probability": round(float(proba[1]), 4),
            "retention_probability": round(float(proba[0]), 4)
        }
