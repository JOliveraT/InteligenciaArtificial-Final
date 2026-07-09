"""Entrena los modelos, ejecuta experimentos y guarda el mejor pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BEST_MODEL_PATH, MODEL_METADATA_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.data_loader import load_dataset
from src.evaluate_models import evaluate_classifier, metrics_table, save_best_model_reports, save_metrics
from src.preprocessing import build_preprocessor, prepare_model_data


def safe_train_test_split(X, y):
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def make_pipeline(X_train, estimator, scale_numeric=True):
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test, specs, experiment_name):
    rows, fitted = [], {}
    for name, estimator, scale_numeric in specs:
        pipe = make_pipeline(X_train, estimator, scale_numeric=scale_numeric)
        pipe.fit(X_train, y_train)
        rows.append(evaluate_classifier(pipe, X_test, y_test, name, experiment_name))
        fitted[name] = pipe
    return metrics_table(rows), fitted


def main():
    raw = load_dataset()
    X, y, processed, feature_columns = prepare_model_data(raw)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    exp1_specs = [
        ("Regresión Logística", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        ("Árbol de Decisión", DecisionTreeClassifier(random_state=RANDOM_STATE), False),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
    ]
    exp2_specs = [
        ("Regresión Logística sin balanceo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        ("Regresión Logística balanced", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), True),
        ("Random Forest sin balanceo", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
        ("Random Forest balanced", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
    ]
    exp3_specs = [
        ("RF básico n100 depthNone", RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1), False),
        ("RF limitado n100 depth10", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1), False),
        ("RF robusto n200 depth20", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1), False),
    ]

    m1, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, exp1_specs, "Experimento 1")
    m2, f2 = train_and_evaluate(X_train, X_test, y_train, y_test, exp2_specs, "Experimento 2")
    m3, f3 = train_and_evaluate(X_train, X_test, y_train, y_test, exp3_specs, "Experimento 3")

    save_metrics(m1, REPORTS_DIR / "metricas_experimento_1.csv")
    save_metrics(m2, REPORTS_DIR / "metricas_experimento_2.csv")
    save_metrics(m3, REPORTS_DIR / "metricas_experimento_3.csv")

    all_metrics = metrics_table(m1.to_dict("records") + m2.to_dict("records") + m3.to_dict("records"))
    save_metrics(all_metrics, REPORTS_DIR / "metricas_todos_los_experimentos.csv")
    fitted = {**f1, **f2, **f3}
    best_name = all_metrics.iloc[0]["modelo"]
    best_model = fitted[best_name]

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    metadata = {
        "best_model_name": best_name,
        "feature_columns": feature_columns,
        "classes": sorted(y.unique().tolist()),
        "n_rows": int(len(processed)),
        "n_features": int(len(feature_columns)),
    }
    joblib.dump(metadata, MODEL_METADATA_PATH)
    labels = [label for label in ["Bajo", "Medio", "Alto"] if label in metadata["classes"]]
    save_best_model_reports(
        best_model, X_test, y_test, labels,
        REPORTS_DIR / "classification_report_mejor_modelo.txt",
        REPORTS_DIR / "matriz_confusion_mejor_modelo.csv",
    )
    print(f"Mejor modelo: {best_name}")
    print(f"Modelo guardado en: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
