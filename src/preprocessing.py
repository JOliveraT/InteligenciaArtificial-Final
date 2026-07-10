"""Funciones de limpieza, ingeniería de variables y pipelines de Scikit-learn."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import EXCLUDED_COLUMNS, FATALITIES_COLUMN, PREFERRED_FEATURES, TARGET_COLUMN


def normalize_column_name(name: object) -> str:
    """Normaliza nombres: mayúsculas, espacios únicos y sin saltos de línea."""
    text = str(name).replace("\n", " ").replace("\r", " ").strip().upper()
    return re.sub(r"\s+", " ", text)


def _without_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def flexible_key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", _without_accents(normalize_column_name(text)))


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def find_column(df: pd.DataFrame, candidates: str | Iterable[str]) -> str | None:
    candidates = [candidates] if isinstance(candidates, str) else list(candidates)
    lookup = {flexible_key(col): col for col in df.columns}
    for candidate in candidates:
        key = flexible_key(candidate)
        if key in lookup:
            return lookup[key]
    return None


def parse_hour(value: object) -> float:
    """Convierte horas tipo time, datetime, 'HH:MM' o número de Excel a hora decimal."""
    if pd.isna(value):
        return np.nan
    if hasattr(value, "hour"):
        return float(value.hour) + float(getattr(value, "minute", 0)) / 60
    if isinstance(value, (int, float)):
        return float(value * 24) if 0 <= float(value) < 1 else float(value)
    text = str(value).strip()
    match = re.search(r"(\d{1,2})(?::(\d{1,2}))?", text)
    if not match:
        return np.nan
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return hour + minute / 60 if 0 <= hour <= 23 and 0 <= minute <= 59 else np.nan


def period_from_hour(hour: object) -> str:
    if pd.isna(hour):
        return "No especificado"
    h = float(hour)
    if 0 <= h < 6:
        return "Madrugada"
    if 6 <= h < 12:
        return "Mañana"
    if 12 <= h < 18:
        return "Tarde"
    if 18 <= h < 24:
        return "Noche"
    return "No especificado"


def create_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea nivel_riesgo y variables derivadas sin filtrar columnas predictoras."""
    out = clean_column_names(df)
    fatalities_col = find_column(out, [FATALITIES_COLUMN, "FALLECIDOS", "CANT FALLECIDOS"])
    if fatalities_col is None:
        raise KeyError(f"Falta la columna obligatoria '{FATALITIES_COLUMN}'.")

    out[fatalities_col] = pd.to_numeric(out[fatalities_col], errors="coerce")
    out = out.dropna(subset=[fatalities_col]).copy()
    out = out[out[fatalities_col] >= 1].copy()

    conditions = [out[fatalities_col].eq(1), out[fatalities_col].eq(2), out[fatalities_col].ge(3)]
    out[TARGET_COLUMN] = np.select(conditions, ["Bajo", "Medio", "Alto"], default="No definido")
    out = out[out[TARGET_COLUMN] != "No definido"].copy()

    date_col = find_column(out, "FECHA SINIESTRO")
    if date_col:
        dates = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
        out["ANIO_SINIESTRO"] = dates.dt.year
        out["MES_SINIESTRO"] = dates.dt.month

    hour_col = find_column(out, "HORA SINIESTRO")
    if hour_col:
        out["HORA_SINIESTRO_NUMERICA"] = out[hour_col].apply(parse_hour)
        out["PERIODO_DIA"] = out["HORA_SINIESTRO_NUMERICA"].apply(period_from_hour)
    return out


def select_feature_columns(df: pd.DataFrame, max_null_ratio: float = 0.70) -> list[str]:
    preferred = [find_column(df, col) for col in PREFERRED_FEATURES]
    derived = ["ANIO_SINIESTRO", "MES_SINIESTRO", "HORA_SINIESTRO_NUMERICA", "PERIODO_DIA"]
    columns = [c for c in preferred + derived if c and c in df.columns]
    excluded_keys = {flexible_key(c) for c in EXCLUDED_COLUMNS}
    clean = []
    for col in dict.fromkeys(columns):
        if flexible_key(col) in excluded_keys:
            continue
        if df[col].isna().mean() <= max_null_ratio:
            clean.append(col)
    if not clean:
        raise ValueError("No se encontraron variables predictoras válidas en el dataset.")
    return clean


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="No especificado")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def prepare_model_data(df: pd.DataFrame):
    processed = create_target_and_features(df)
    feature_columns = select_feature_columns(processed)
    X = processed[feature_columns].copy()
    y = processed[TARGET_COLUMN].copy()
    return X, y, processed, feature_columns
