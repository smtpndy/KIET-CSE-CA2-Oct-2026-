
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes risk using a trained ML model",
    version="1.0.0",
)

# Load the saved model artifact
MODEL_PATH = Path(__file__).parent / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

# Supports both the dictionary artifact and a directly saved model
if isinstance(artifact, dict):
    model = artifact["model"]
    feature_names = artifact.get(
        "feature_names",
        [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ],
    )
else:
    model = artifact
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ]


# Request schema
class PatientData(BaseModel):
    Pregnancies: int = Field(ge=0, le=30)
    Glucose: float = Field(ge=0, le=500)
    BloodPressure: float = Field(ge=0, le=300)
    SkinThickness: float = Field(ge=0, le=150)
    Insulin: float = Field(ge=0, le=3000)
    BMI: float = Field(ge=0, le=150)
    DiabetesPedigreeFunction: float = Field(ge=0, le=10)
    Age: int = Field(ge=1, le=120)


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PatientData):
    try:
        # Arrange input columns in the exact training order
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=feature_names,
        )

        prediction = int(model.predict(input_data)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        return {
            "prediction": prediction,
            "result": (
                "Diabetes predicted"
                if prediction == 1
                else "No diabetes predicted"
            ),
            "probability_of_class_1": probability,
            "note": "Educational prediction only; not a medical diagnosis.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
