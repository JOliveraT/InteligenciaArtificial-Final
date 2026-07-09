"""Utilidades de predicción para la aplicación Streamlit."""
from pathlib import Path
import joblib
import pandas as pd
from .config import BEST_MODEL_PATH, MODEL_METADATA_PATH


def load_model(model_path: Path = BEST_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Primero ejecute: python src/train_models.py"
        )
    return joblib.load(model_path)


def load_metadata(path: Path = MODEL_METADATA_PATH) -> dict:
    return joblib.load(path) if path.exists() else {}


def predict_single(input_data: dict, model=None) -> tuple[str, dict[str, float]]:
    model = model or load_model()
    X = pd.DataFrame([input_data])
    prediction = model.predict(X)[0]
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(X)[0].round(4)))
    return prediction, probabilities
