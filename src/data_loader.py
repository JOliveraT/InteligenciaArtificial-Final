"""Carga y validación del dataset Excel del ONSV."""
from pathlib import Path
import pandas as pd
from .config import EXCEL_PATH, SHEET_NAME, FATALITIES_COLUMN

EXCEL_HEADER_ROW = 3
REQUIRED_COLUMNS = {
    "CÓDIGO SINIESTRO",
    "FECHA SINIESTRO",
    FATALITIES_COLUMN,
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
}


class DatasetNotFoundError(FileNotFoundError):
    """Error claro cuando el Excel no existe."""


def validate_dataset_path(path: Path = EXCEL_PATH) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"No se encontró el dataset en {path}. Coloque el Excel del ONSV en esa ruta exacta."
        )
    return path


def validate_dataset_columns(df: pd.DataFrame) -> None:
    """Verifica que el Excel se haya leído con los encabezados esperados."""
    if FATALITIES_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{FATALITIES_COLUMN}' después de cargar el Excel. "
            "Probablemente el encabezado del Excel cambió o debe revisarse el parámetro "
            f"header={EXCEL_HEADER_ROW}."
        )

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "El Excel se cargó, pero faltan columnas esperadas del ONSV: "
            f"{missing}. Revise si el encabezado del Excel cambió o si debe ajustarse "
            f"header={EXCEL_HEADER_ROW}."
        )


def load_dataset(path: Path = EXCEL_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Lee la hoja SINIESTROS del Excel oficial usando la fila real de encabezados."""
    dataset_path = validate_dataset_path(path)
    try:
        df = pd.read_excel(
            dataset_path,
            sheet_name=sheet_name,
            header=EXCEL_HEADER_ROW,
            engine="openpyxl",
        )
    except ValueError as exc:
        raise ValueError(f"No se pudo leer la hoja '{sheet_name}' en {dataset_path}.") from exc

    validate_dataset_columns(df)
    return df
