# """
# main.py — FastAPI service for the Customer Churn prediction model.

# Loads a single joblib file containing the full trained pipeline
# (preprocessing + CatBoost classifier).

# Run locally with:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Deployed on Render with:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# """

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Request schema — RAW fields only, exactly what a user/frontend submits.
#    age_group / tenure_group / has_zero_balance / inactive_with_one_product
#    are all computed automatically below — never asked from the user.
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=900, example=650)
    country: str = Field(..., example="France")            # "France" | "Germany" | "Spain"
    gender: str = Field(..., example="Female")              # "Male" | "Female"
    age: int = Field(..., ge=18, le=100, example=42)
    tenure: int = Field(..., ge=0, le=15, example=3)
    balance: float = Field(..., ge=0, example=83807.86)
    products_number: int = Field(..., ge=1, le=4, example=1)
    credit_card: int = Field(..., ge=0, le=1, example=1)        # 0 = no, 1 = yes
    active_member: int = Field(..., ge=0, le=1, example=1)      # 0 = no, 1 = yes
    estimated_salary: float = Field(..., ge=0, example=101348.88)


class ChurnPrediction(BaseModel):
    churn_prediction: str          # "churn" | "not churn"
    churn_probability: float       # 0-1
    retention_probability: float   # 0-1


# ---------------------------------------------------------------------------
# 2. App setup + CORS (needed so a Lovable-hosted frontend can call this API
#    from a different domain)
# ---------------------------------------------------------------------------
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your actual frontend domain once live
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 3. Load the trained pipeline
# ---------------------------------------------------------------------------
model = joblib.load("model/best_churn_pipeline.joblib")


# ---------------------------------------------------------------------------
# 4. Helpers — bin raw age/tenure the SAME way they were binned in training
# ---------------------------------------------------------------------------
def make_age_group(age: int) -> str:
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    return str(pd.cut([age], bins=bins, labels=labels)[0])


def make_tenure_group(tenure: int) -> str:
    bins = [-1, 0, 2, 5, 10, 100]
    labels = ['0', '1-2', '3-5', '6-10', '10+']
    return str(pd.cut([tenure], bins=bins, labels=labels)[0])


def build_feature_row(payload: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "credit_score": payload.credit_score,
        "country": payload.country,
        "gender": payload.gender,
        "age_group": make_age_group(payload.age),
        "tenure_group": make_tenure_group(payload.tenure),
        "balance": payload.balance,
        "products_number": payload.products_number,
        "credit_card": "yes" if payload.credit_card == 1 else "no",
        "active_member": "yes" if payload.active_member == 1 else "no",
        "estimated_salary": payload.estimated_salary,
        "has_zero_balance": "yes" if payload.balance == 0 else "no",
        "inactive_with_one_product": "yes" if (
            payload.active_member == 0 and payload.products_number == 1
        ) else "no",
    }])


# ---------------------------------------------------------------------------
# 5. Routes
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn prediction API is running."}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(payload: CustomerInput):
    try:
        row_df = build_feature_row(payload)

        prediction = model.predict(row_df)[0]
        proba = model.predict_proba(row_df)[0]   # [P(not churn), P(churn)]

        return ChurnPrediction(
            churn_prediction="churn" if prediction == 1 else "not churn",
            churn_probability=round(float(proba[1]), 4),
            retention_probability=round(float(proba[0]), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
