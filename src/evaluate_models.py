"""Evaluación y persistencia de métricas de clasificación."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def evaluate_classifier(model, X_test, y_test, model_name: str, experiment: str) -> dict:
    y_pred = model.predict(X_test)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    return {
        "experimento": experiment,
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["f1_macro", "f1_weighted"], ascending=False)


def save_metrics(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_best_model_reports(model, X_test, y_test, labels: list[str], report_path: Path, matrix_path: Path) -> None:
    y_pred = model.predict(X_test)
    report_path.write_text(
        classification_report(y_test, y_pred, labels=labels, zero_division=0), encoding="utf-8"
    )
    matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels)
    matrix.to_csv(matrix_path, encoding="utf-8-sig")
