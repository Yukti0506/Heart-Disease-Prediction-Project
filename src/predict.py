"""
predict.py
----------
Helper functions for loading the saved model pipeline and metadata, and for
turning a single patient's raw form input into a prediction. Used by the
Streamlit app (app/app.py) and can also be used standalone / from other
scripts.
"""

import json
import os

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "heart_disease_model.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")


def load_model(path: str = MODEL_PATH):
    """Load the saved full pipeline (preprocessing + trained classifier)."""
    return joblib.load(path)


def load_metadata(path: str = METADATA_PATH) -> dict:
    """Load feature names, category options and numeric ranges."""
    with open(path, "r") as f:
        return json.load(f)


def predict_single(pipeline, patient_dict: dict):
    """
    Run a prediction for a single patient.

    Parameters
    ----------
    pipeline : the fitted sklearn Pipeline (preprocessing + classifier)
    patient_dict : dict mapping raw feature names (age, sex, cp, trestbps,
        chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, dataset)
        to raw values, in the same format the model was trained on.

    Returns
    -------
    (prediction:int, probability_of_disease:float)
    """
    X_new = pd.DataFrame([patient_dict])
    prediction = int(pipeline.predict(X_new)[0])
    probability = float(pipeline.predict_proba(X_new)[0][1])
    return prediction, probability
