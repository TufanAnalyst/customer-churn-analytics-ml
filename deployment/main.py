from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

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

    if prediction == 1:
        return {"result": "Churn"}
    else:
        return {"result": "Not Churn"}
