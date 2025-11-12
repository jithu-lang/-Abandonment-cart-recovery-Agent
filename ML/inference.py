import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
META_PATH = os.path.join(BASE_DIR, "feature_metadata.joblib")

try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)

    FEATURES = metadata["features"]
    CATEGORICAL = metadata["categorical"]

    print(" Loaded advanced ML model (CatBoost).")

except Exception as e:
    print(" Failed to load advanced model:", e)
    model = None
    FEATURES = []
    CATEGORICAL = []


def predict_recovery_prob_advanced(feature_dict: dict) -> float:
    if model is None:
        print("Model not loaded. Returning fallback ML score = 50.0")
        return 50.0

    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=FEATURES, fill_value=0)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        print(" Prediction failed:", e)
        return 50.0

    return round(float(prob * 100), 2) 
