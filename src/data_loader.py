"""Carga y validación del dataset Excel del ONSV."""
from pathlib import Path
import pandas as pd
from .config import EXCEL_PATH, SHEET_NAME

class DatasetNotFoundError(FileNotFoundError):
    """Error claro cuando el Excel no existe."""


def validate_dataset_path(path: Path = EXCEL_PATH) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"No se encontró el dataset en {path}. Coloque el Excel del ONSV en esa ruta exacta."
        )
    return path


def load_dataset(path: Path = EXCEL_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Lee la hoja SINIESTROS del Excel oficial."""
    dataset_path = validate_dataset_path(path)
    try:
        return pd.read_excel(dataset_path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(f"No se pudo leer la hoja '{sheet_name}' en {dataset_path}.") from exc
