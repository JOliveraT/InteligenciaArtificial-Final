"""Aplicación Streamlit del proyecto final de IA."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import BEST_MODEL_PATH, EXCEL_PATH, REPORTS_DIR, TARGET_COLUMN
from src.data_loader import DatasetNotFoundError, load_dataset
from src.predict import load_metadata, load_model, predict_single
from src.preprocessing import prepare_model_data

st.set_page_config(page_title="IA Siniestros Fatales Perú", page_icon="🚦", layout="wide")

@st.cache_data(show_spinner=False)
def get_data():
    raw = load_dataset()
    X, y, processed, feature_columns = prepare_model_data(raw)
    return raw, X, y, processed, feature_columns

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


FRIENDLY_FEATURE_LABELS = {
    "CLASE SINIESTRO": "Clase de siniestro",
    "CANTIDAD DE LESIONADOS": "Cantidad de lesionados",
    "CANTIDAD DE VEHICULOS DAÑADOS": "Cantidad de vehículos dañados",
    "DEPARTAMENTO": "Departamento",
    "PROVINCIA": "Provincia",
    "DISTRITO": "Distrito",
    "ZONA": "Zona",
    "TIPO DE VÍA": "Tipo de vía",
    "RED VIAL": "Red vial",
    "COD CARRETERA": "Código de carretera",
    "CONDICIÓN CLIMÁTICA": "Condición climática",
    "ZONIFICACIÓN": "Zonificación",
    "CARACTERÍSTICAS DE VÍA": "Características de vía",
    "PERFIL LONGITUDINAL VÍA": "Perfil longitudinal de vía",
    "SUPERFICIE DE CALZADA": "Superficie de calzada",
    "CAUSA FACTOR PRINCIPAL": "Causa / factor principal",
    "CAUSA ESPECÍFICA": "Causa específica",
    "anio_siniestro": "Año del siniestro",
    "mes_siniestro": "Mes del siniestro",
    "hora_siniestro_numerica": "Hora del siniestro",
    "periodo_dia": "Periodo del día",
}

MONTH_OPTIONS = {
    1: "1 - Enero",
    2: "2 - Febrero",
    3: "3 - Marzo",
    4: "4 - Abril",
    5: "5 - Mayo",
    6: "6 - Junio",
    7: "7 - Julio",
    8: "8 - Agosto",
    9: "9 - Septiembre",
    10: "10 - Octubre",
    11: "11 - Noviembre",
    12: "12 - Diciembre",
}


def friendly_label(feature):
    """Etiqueta legible para la UI sin modificar el nombre real de la feature."""
    return FRIENDLY_FEATURE_LABELS.get(feature, feature)


def sort_text_options(options):
    """Ordena textos y mantiene 'No especificado' al final de forma consistente."""
    text_options = [str(option) for option in options]
    clean_options = [option for option in text_options if option != "No especificado"]
    sorted_options = sorted(clean_options)
    return sorted_options + (["No especificado"] if "No especificado" in text_options else [])


def get_integer_options(df, column, fallback_options):
    """Devuelve valores enteros únicos de una columna numérica con fallback seguro."""
    if column not in df.columns:
        return fallback_options
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return fallback_options
    options = sorted(values.astype(int).unique().tolist())
    return options or fallback_options


def get_default_year(df):
    """Prefiere 2024 si existe; si no, usa el año más frecuente del dataset."""
    years = get_integer_options(df, "anio_siniestro", [2021, 2022, 2023, 2024, 2025])
    if 2024 in years:
        return 2024
    if "anio_siniestro" not in df.columns:
        return years[-1]
    mode = pd.to_numeric(df["anio_siniestro"], errors="coerce").dropna().astype(int).mode()
    return int(mode.iloc[0]) if not mode.empty else years[-1]


def calculate_period(hour):
    """Calcula el periodo del día a partir de la hora del siniestro."""
    if 0 <= hour <= 5:
        return "Madrugada"
    if 6 <= hour <= 11:
        return "Mañana"
    if 12 <= hour <= 17:
        return "Tarde"
    return "Noche"


def bar_count(df, column, title, top=15):
    if column not in df.columns:
        st.info(f"La columna {column} no está disponible en el dataset.")
        return
    counts = df[column].fillna("No especificado").value_counts().head(top).reset_index()
    counts.columns = [column, "cantidad"]
    st.plotly_chart(px.bar(counts, x=column, y="cantidad", title=title), width="stretch")


def get_text_options(df, column, limit=200):
    """Devuelve opciones de texto limpias para un selectbox con fallback seguro."""
    if column not in df.columns:
        return ["No especificado"]
    options = sort_text_options(df[column].fillna("No especificado").astype(str).unique().tolist())
    return options[:limit] or ["No especificado"]


def get_filtered_text_options(df, column, filters, fallback_df=None, limit=200):
    """Filtra opciones por columnas relacionadas y usa todos los valores como fallback."""
    if column not in df.columns:
        return ["No especificado"]

    filtered = df.copy()
    for filter_column, selected_value in filters.items():
        if filter_column not in filtered.columns or selected_value is None:
            continue
        filtered = filtered[
            filtered[filter_column].fillna("No especificado").astype(str) == str(selected_value)
        ]

    if filtered.empty:
        filtered = fallback_df if fallback_df is not None else df

    return get_text_options(filtered, column, limit=limit)


st.title("🚦 Clasificación del nivel de riesgo en siniestros de tránsito fatales en el Perú")
st.caption("Proyecto Final de Inteligencia Artificial · Ingeniería de Software · Universidad La Salle")

st.markdown(
    """
Esta aplicación usa Machine Learning supervisado para clasificar el **nivel de riesgo** de siniestros fatales
registrados por el Observatorio Nacional de Seguridad Vial del Perú (ONSV). Como todos los registros son fatales,
el objetivo no es predecir fatalidad, sino clasificar el riesgo según fallecidos: **Bajo** (1), **Medio** (2) y **Alto** (3+).
"""
)

try:
    raw_df, X, y, df, feature_columns = get_data()
except DatasetNotFoundError as exc:
    st.error(str(exc))
    st.info(f"Coloque el Excel en: `{EXCEL_PATH}` y vuelva a ejecutar la app.")
    st.stop()
except Exception as exc:
    st.error(f"No se pudo preparar el dataset: {exc}")
    st.stop()

tab_intro, tab_eda, tab_exp, tab_pred = st.tabs(["📌 Resumen", "📊 EDA", "🧪 Experimentos", "🔮 Predicción"])

with tab_intro:
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Variables originales", raw_df.shape[1])
    c3.metric("Variables predictoras", len(feature_columns))
    st.subheader("Fuente de datos")
    st.write("Fuente: Observatorio Nacional de Seguridad Vial del Perú, ONSV. Dataset: BBDD ONSV - Siniestros Fatales 2021-2025 preliminar.")
    st.subheader("Pipeline")
    st.code("""Dataset ONSV Excel → Carga → Limpieza → EDA → nivel_riesgo → Preprocesamiento
→ Train/test split → Modelos Scikit-learn → Evaluación → Mejor modelo → Streamlit → Predicción""")
    st.write("Variables usadas por el modelo:", ", ".join(feature_columns))

with tab_eda:
    st.subheader("Vista previa de datos procesados")
    st.dataframe(df.head(50), width="stretch")
    st.subheader("Distribución del nivel de riesgo")
    risk_counts = df[TARGET_COLUMN].value_counts().reset_index()
    risk_counts.columns = ["nivel_riesgo", "cantidad"]
    st.plotly_chart(px.pie(risk_counts, names="nivel_riesgo", values="cantidad", title="Distribución de nivel_riesgo"), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        bar_count(df, "DEPARTAMENTO", "Siniestros por departamento")
        bar_count(df, "TIPO DE VÍA", "Siniestros por tipo de vía")
    with col_b:
        bar_count(df, "CLASE SINIESTRO", "Siniestros por clase de siniestro")
        bar_count(df, "ZONA", "Siniestros por zona")
    st.subheader("Estadísticas generales")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")

with tab_exp:
    st.subheader("Resultados de experimentos")
    metric_files = sorted(REPORTS_DIR.glob("metricas_experimento_*.csv"))
    if not metric_files:
        st.warning("Aún no existen resultados. Ejecute primero: `python src/train_models.py`")
    else:
        metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
        st.dataframe(metrics, width="stretch")
        metric = st.selectbox("Métrica para comparar", ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"], index=3)
        st.plotly_chart(px.bar(metrics, x="modelo", y=metric, color="experimento", barmode="group", title=f"Comparación por {metric}"), width="stretch")
        best = metrics.sort_values(["f1_macro", "f1_weighted"], ascending=False).iloc[0]
        st.success(f"Mejor resultado observado: {best['modelo']} con F1 macro = {best['f1_macro']:.3f}.")
    matrix_path = REPORTS_DIR / "matriz_confusion_mejor_modelo.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        st.plotly_chart(px.imshow(matrix, text_auto=True, title="Matriz de confusión del mejor modelo"), width="stretch")

with tab_pred:
    st.subheader("Formulario de predicción")
    if not BEST_MODEL_PATH.exists():
        st.warning("No existe el modelo entrenado. Primero ejecute: `python src/train_models.py`")
        st.stop()
    model = get_model()
    metadata = load_metadata()
    user_input = {}

    prediction_features = metadata.get("feature_columns", feature_columns)
    has_location_fields = all(
        feature in prediction_features and feature in X.columns
        for feature in ["DEPARTAMENTO", "PROVINCIA", "DISTRITO"]
    )

    if has_location_fields:
        st.markdown("**Ubicación del siniestro**")
        location_cols = st.columns(3)
        with location_cols[0]:
            department_options = get_text_options(X, "DEPARTAMENTO")
            user_input["DEPARTAMENTO"] = st.selectbox(friendly_label("DEPARTAMENTO"), department_options)
        with location_cols[1]:
            province_options = get_filtered_text_options(
                X,
                "PROVINCIA",
                {"DEPARTAMENTO": user_input["DEPARTAMENTO"]},
                fallback_df=X,
            )
            user_input["PROVINCIA"] = st.selectbox(friendly_label("PROVINCIA"), province_options)
        with location_cols[2]:
            district_options = get_filtered_text_options(
                X,
                "DISTRITO",
                {
                    "DEPARTAMENTO": user_input["DEPARTAMENTO"],
                    "PROVINCIA": user_input["PROVINCIA"],
                },
                fallback_df=X,
            )
            user_input["DISTRITO"] = st.selectbox(friendly_label("DISTRITO"), district_options)

    with st.form("prediction_form"):
        cols = st.columns(2)
        form_features = [
            feature
            for feature in prediction_features
            if not (has_location_fields and feature in ["DEPARTAMENTO", "PROVINCIA", "DISTRITO"])
        ]
        skip_features = {"COD CARRETERA", "periodo_dia"}
        for i, feature in enumerate([item for item in form_features if item not in skip_features]):
            series = X[feature] if feature in X.columns else pd.Series(dtype=object)
            with cols[i % 2]:
                label = friendly_label(feature)
                if feature == "CANTIDAD DE LESIONADOS":
                    user_input[feature] = st.number_input(label, min_value=0, step=1, value=0)
                elif feature == "CANTIDAD DE VEHICULOS DAÑADOS":
                    user_input[feature] = st.number_input(label, min_value=0, step=1, value=1)
                elif feature == "anio_siniestro":
                    year_options = get_integer_options(X, feature, [2021, 2022, 2023, 2024, 2025])
                    default_year = get_default_year(X)
                    default_index = year_options.index(default_year) if default_year in year_options else 0
                    user_input[feature] = st.selectbox(label, year_options, index=default_index)
                elif feature == "mes_siniestro":
                    user_input[feature] = st.selectbox(
                        label,
                        list(MONTH_OPTIONS.keys()),
                        index=0,
                        format_func=lambda month: MONTH_OPTIONS[month],
                    )
                elif feature == "hora_siniestro_numerica":
                    user_input[feature] = st.slider(label, min_value=0, max_value=23, value=12, step=1)
                    user_input["periodo_dia"] = calculate_period(user_input[feature])
                    st.info(f"Periodo del día calculado: {user_input['periodo_dia']}")
                elif pd.api.types.is_numeric_dtype(series):
                    default = float(series.median()) if not series.dropna().empty else 0.0
                    user_input[feature] = st.number_input(label, value=default)
                else:
                    options = get_text_options(X, feature) if feature in X.columns else ["No especificado"]
                    user_input[feature] = st.selectbox(label, options or ["No especificado"])

        if "COD CARRETERA" in form_features:
            road_code_options = get_filtered_text_options(
                X,
                "COD CARRETERA",
                {"RED VIAL": user_input.get("RED VIAL")},
                fallback_df=X,
            )
            user_input["COD CARRETERA"] = st.selectbox(
                friendly_label("COD CARRETERA"),
                road_code_options or ["No especificado"],
            )

        if "periodo_dia" in form_features and "periodo_dia" not in user_input:
            hour = int(user_input.get("hora_siniestro_numerica", 12))
            user_input["periodo_dia"] = calculate_period(hour)
            st.info(f"Periodo del día calculado: {user_input['periodo_dia']}")

        submitted = st.form_submit_button("Predecir nivel de riesgo")
    if submitted:
        prediction, probabilities = predict_single(user_input, model=model)
        st.success(f"Resultado: Riesgo {prediction}")
        if probabilities:
            prob_df = pd.DataFrame({"nivel_riesgo": probabilities.keys(), "probabilidad": probabilities.values()})
            st.plotly_chart(px.bar(prob_df, x="nivel_riesgo", y="probabilidad", title="Probabilidades estimadas"), width="stretch")
        st.info("Interpretación: el resultado resume el patrón aprendido en registros históricos; no reemplaza análisis pericial ni políticas públicas de seguridad vial.")
