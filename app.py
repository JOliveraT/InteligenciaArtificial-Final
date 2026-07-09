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


def bar_count(df, column, title, top=15):
    if column not in df.columns:
        st.info(f"La columna {column} no está disponible en el dataset.")
        return
    counts = df[column].fillna("No especificado").value_counts().head(top).reset_index()
    counts.columns = [column, "cantidad"]
    st.plotly_chart(px.bar(counts, x=column, y="cantidad", title=title), use_container_width=True)


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
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Distribución del nivel de riesgo")
    risk_counts = df[TARGET_COLUMN].value_counts().reset_index()
    risk_counts.columns = ["nivel_riesgo", "cantidad"]
    st.plotly_chart(px.pie(risk_counts, names="nivel_riesgo", values="cantidad", title="Distribución de nivel_riesgo"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        bar_count(df, "DEPARTAMENTO", "Siniestros por departamento")
        bar_count(df, "TIPO DE VÍA", "Siniestros por tipo de vía")
    with col_b:
        bar_count(df, "CLASE SINIESTRO", "Siniestros por clase de siniestro")
        bar_count(df, "ZONA", "Siniestros por zona")
    st.subheader("Estadísticas generales")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with tab_exp:
    st.subheader("Resultados de experimentos")
    metric_files = sorted(REPORTS_DIR.glob("metricas_experimento_*.csv"))
    if not metric_files:
        st.warning("Aún no existen resultados. Ejecute primero: `python src/train_models.py`")
    else:
        metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
        st.dataframe(metrics, use_container_width=True)
        metric = st.selectbox("Métrica para comparar", ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"], index=3)
        st.plotly_chart(px.bar(metrics, x="modelo", y=metric, color="experimento", barmode="group", title=f"Comparación por {metric}"), use_container_width=True)
        best = metrics.sort_values(["f1_macro", "f1_weighted"], ascending=False).iloc[0]
        st.success(f"Mejor resultado observado: {best['modelo']} con F1 macro = {best['f1_macro']:.3f}.")
    matrix_path = REPORTS_DIR / "matriz_confusion_mejor_modelo.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        st.plotly_chart(px.imshow(matrix, text_auto=True, title="Matriz de confusión del mejor modelo"), use_container_width=True)

with tab_pred:
    st.subheader("Formulario de predicción")
    if not BEST_MODEL_PATH.exists():
        st.warning("No existe el modelo entrenado. Primero ejecute: `python src/train_models.py`")
        st.stop()
    model = get_model()
    metadata = load_metadata()
    user_input = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, feature in enumerate(metadata.get("feature_columns", feature_columns)):
            series = X[feature] if feature in X.columns else pd.Series(dtype=object)
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    default = float(series.median()) if not series.dropna().empty else 0.0
                    user_input[feature] = st.number_input(feature, value=default)
                else:
                    options = sorted(series.fillna("No especificado").astype(str).unique().tolist())[:200]
                    user_input[feature] = st.selectbox(feature, options or ["No especificado"])
        submitted = st.form_submit_button("Predecir nivel de riesgo")
    if submitted:
        prediction, probabilities = predict_single(user_input, model=model)
        st.success(f"Resultado: Riesgo {prediction}")
        if probabilities:
            prob_df = pd.DataFrame({"nivel_riesgo": probabilities.keys(), "probabilidad": probabilities.values()})
            st.plotly_chart(px.bar(prob_df, x="nivel_riesgo", y="probabilidad", title="Probabilidades estimadas"), use_container_width=True)
        st.info("Interpretación: el resultado resume el patrón aprendido en registros históricos; no reemplaza análisis pericial ni políticas públicas de seguridad vial.")
