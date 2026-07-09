# Proyecto Final de Inteligencia Artificial

## 1. Título
Aplicación web para la clasificación del nivel de riesgo en siniestros de tránsito fatales en el Perú mediante Machine Learning.

## 2. Integrantes
- Integrante 1: ______________________________
- Integrante 2: ______________________________
- Integrante 3: ______________________________

## 3. Problema
Los siniestros de tránsito fatales representan un problema importante de seguridad vial en el Perú. Este proyecto busca clasificar el nivel de riesgo de un siniestro fatal según sus características registradas, para apoyar el análisis exploratorio y la toma de decisiones basada en datos.

## 4. Objetivo
Desarrollar una aplicación web funcional que use Machine Learning para clasificar el nivel de riesgo de siniestros de tránsito fatales en Perú.

## 5. Fuente de datos
- Fuente: Observatorio Nacional de Seguridad Vial, ONSV.
- Dataset: BBDD ONSV - Siniestros Fatales 2021-2025 preliminar.
- Formato: Excel.
- Hoja usada: SINIESTROS.
- Periodo: 2021-2025 preliminar.
- Archivo esperado: `data/BBDD ONSV - SINIESTROS FATALES 2021-2025 (preliminar).xlsx`.
- Fecha de obtención: ______________________________.
- Licencia: debe verificarse en el portal de datos abiertos del ONSV.

> Si el archivo Excel aún no está en el repositorio, colóquelo manualmente en la ruta exacta indicada antes de entrenar o ejecutar la aplicación.

## 6. Técnica de IA utilizada
Se utiliza Machine Learning supervisado con Scikit-learn porque el dataset es tabular y el objetivo es un problema de clasificación multiclase.

Modelos comparados:
- Regresión Logística.
- Árbol de Decisión.
- Random Forest.

La elección permite comparar un modelo lineal e interpretable, un modelo basado en reglas y un ensamble más robusto para relaciones no lineales.

## 7. Variable objetivo
La variable `nivel_riesgo` se crea desde `CANTIDAD DE FALLECIDOS`:
- 1 fallecido = Bajo.
- 2 fallecidos = Medio.
- 3 o más fallecidos = Alto.

La columna `CANTIDAD DE FALLECIDOS` no se usa como variable predictora.

## 8. Pipeline
```text
Dataset ONSV Excel
        ↓
Carga de datos
        ↓
Limpieza y normalización de columnas
        ↓
EDA
        ↓
Creación de variable objetivo nivel_riesgo
        ↓
Preprocesamiento
        ↓
División train/test
        ↓
Entrenamiento de modelos
        ↓
Evaluación
        ↓
Selección del mejor modelo
        ↓
Aplicación Streamlit
        ↓
Predicción e interpretación
```

## 9. Experimentos
1. Comparación de modelos base: Regresión Logística, Árbol de Decisión y Random Forest.
2. Comparación con y sin `class_weight="balanced"`: evalúa el efecto del balanceo de clases.
3. Ajuste de hiperparámetros de Random Forest: compara profundidad y número de árboles.

## 10. Métricas
Se calculan:
- Accuracy.
- Precision macro.
- Recall macro.
- F1 macro.
- F1 weighted.
- Matriz de confusión.

F1 macro y F1 weighted son importantes porque la clase Bajo puede dominar el dataset, generando desbalance.

## 11. Resultados
Después de ejecutar el entrenamiento se generan archivos en `reports/`:

| Archivo | Descripción |
| --- | --- |
| `metricas_experimento_1.csv` | Comparación de modelos base |
| `metricas_experimento_2.csv` | Comparación con y sin balanceo |
| `metricas_experimento_3.csv` | Hiperparámetros de Random Forest |
| `classification_report_mejor_modelo.txt` | Reporte del mejor modelo |
| `matriz_confusion_mejor_modelo.csv` | Matriz de confusión |

## 12. Aplicación
La aplicación Streamlit permite:
- Ver la descripción del problema y el pipeline.
- Visualizar datos y gráficos EDA.
- Consultar resultados de experimentos.
- Ingresar características de un siniestro.
- Obtener el nivel de riesgo predicho y probabilidades estimadas.

## 13. Instalación
Comandos para Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 14. Ejecución del entrenamiento
```powershell
python src/train_models.py
```

## 15. Ejecución de la aplicación
```powershell
streamlit run app.py
```

## 16. Estructura del repositorio
```text
proyecto-final-ia-siniestros/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_eda_modelado.ipynb
├── reports/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train_models.py
    ├── evaluate_models.py
    └── predict.py
```

## 17. Notas
- El archivo Excel debe colocarse en la carpeta `data/`.
- Si el modelo no existe, primero se debe ejecutar `python src/train_models.py`.
- El año 2025 es preliminar, por lo que sus registros pueden cambiar.
- El sistema es académico y no reemplaza análisis técnico especializado de seguridad vial.
