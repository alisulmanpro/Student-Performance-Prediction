from fastapi import FastAPI
import joblib
import pandas as pd

description = """
### Overview
The **Student Performance Prediction API** leverages a Random Forest model to predict academic outcomes based on demographic and behavioral data.

### Key Features
* **Performance Prediction:** Estimates final grades or pass/fail status.
* **Feature Analysis:** Processes input variables like study hours, attendance, and previous scores.
* **Fast & Scalable:** Built on FastAPI for high-performance inference.

### How to Use
1.  **POST** your student data to the `/predict` endpoint.
2.  Receive a JSON response with the predicted outcome.
"""

app = FastAPI(
    title="Student Performance Prediction API",
    description=description,
    version="1.0.0",
)

MODEL_PATH = "models/student_performance_rf.pkl"
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON with feature names as keys
    """
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {
        "predicted_G3": round(float(prediction), 2)
    }
