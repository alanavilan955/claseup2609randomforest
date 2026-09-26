"""
App de Streamlit: predicción de tensión de liquidez con Random Forest.

Configuración aprobada:
- Objetivo: clasificación binaria de Tension_Liquidez_bin.
- 21 predictores del Diccionario (Sector + 19 variables numéricas + Estacionalidad_bin).
- Excluidos del entrenamiento: ID_Observacion, Fecha, Prob_Tension_Liquidez (leakage),
  Brecha_Caja_90d_MXN (target del modelo de regresión, no se usa aquí).
- Sector: One-Hot Encoding.
- Split 70/30, random_state=42, estratificado por el target.
- RandomForestClassifier con class_weight='balanced'.
- Hiperparámetros fijos: n_estimators=300, max_depth=None, min_samples_split=2,
  min_samples_leaf=1, max_features='sqrt', random_state=42.
- Umbral de clasificación fijo en 0.5.
"""

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Tensión de liquidez — Random Forest", layout="wide")

TARGET = "Tension_Liquidez_bin"

CATEGORICAL_FEATURES = ["Sector"]

NUMERIC_FEATURES = [
    "Ventas_12M_MXN",
    "Crecimiento_Ventas_pct",
    "Margen_EBITDA_pct",
    "DSO_dias",
    "DIO_dias",
    "DPO_dias",
    "CCC_dias",
    "CxC_MXN",
    "Inventarios_MXN",
    "CxP_MXN",
    "NWC_MXN",
    "Deuda_CP_MXN",
    "Caja_MXN",
    "Linea_Credito_Disponible_MXN",
    "Concentracion_Top5_Clientes_pct",
    "Morosidad_CxC_pct",
    "Inventario_Obsoleto_pct",
    "Volatilidad_Ventas_pct",
    "Estacionalidad_bin",
    "EBITDA_MXN",
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLS = FEATURE_COLS + [TARGET]

DEFAULT_DATA_PATH = "Base_Didactica_Random_Forest_Capital_Trabajo.xlsx"
DEFAULT_SHEET = "Datos_Modelo"


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_DATA_PATH, sheet_name=DEFAULT_SHEET)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    xls = pd.ExcelFile(uploaded_file)
    sheet = DEFAULT_SHEET if DEFAULT_SHEET in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


def validate_columns(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_COLS if c not in df.columns]


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[("sector", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    ohe = pipeline.named_steps["preprocessor"].named_transformers_["sector"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = cat_names + NUMERIC_FEATURES
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"variable": all_feature_names, "importancia": importances})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "importance_df": importance_df,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title("Predicción de tensión de liquidez — Random Forest")
st.caption(
    "Clasificación binaria sobre variables de capital de trabajo. "
    "Target: Tension_Liquidez_bin (1 = tensión de liquidez, 0 = situación normal)."
)

st.sidebar.header("Datos de entrada")
uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo propio (.xlsx o .csv) con la misma estructura de columnas",
    type=["xlsx", "csv"],
)

if uploaded_file is not None:
    df_raw = read_uploaded_file(uploaded_file)
    fuente = f"Archivo cargado: {uploaded_file.name}"
else:
    df_raw = load_default_data()
    fuente = "Base didáctica por defecto (Base_Didactica_Random_Forest_Capital_Trabajo.xlsx)"

st.sidebar.caption(fuente)

missing = validate_columns(df_raw)
if missing:
    st.error(
        "El archivo cargado no tiene la estructura esperada. Faltan las columnas: "
        + ", ".join(missing)
    )
    st.stop()

df = df_raw.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
descartadas = len(df_raw) - len(df)

result = train_model(df)

tab_datos, tab_metricas, tab_importancia = st.tabs(
    ["Datos", "Métricas y matriz de confusión", "Importancia de variables"]
)

with tab_datos:
    st.subheader("Vista previa de los datos")
    st.write(
        f"Observaciones usadas: {len(df)} "
        f"(descartadas por valores nulos: {descartadas}) · "
        f"Entrenamiento: {result['n_train']} · Prueba: {result['n_test']}"
    )
    st.dataframe(df[FEATURE_COLS + [TARGET]].head(20), use_container_width=True)

    st.subheader("Distribución del target")
    dist = df[TARGET].value_counts(normalize=True).sort_index()
    dist.index = ["0 — normal", "1 — tensión de liquidez"]
    st.bar_chart(dist.rename("proporción"))

with tab_metricas:
    st.subheader("Métricas sobre el conjunto de prueba (30%, umbral = 0.5)")
    m = result["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']:.3f}")
    col2.metric("Precision", f"{m['precision']:.3f}")
    col3.metric("Recall", f"{m['recall']:.3f}")
    col4.metric("F1", f"{m['f1']:.3f}")
    col5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("**Matriz de confusión**")
        cm = result["cm"]
        cm_df = pd.DataFrame(
            cm,
            index=["Real 0 (normal)", "Real 1 (tensión)"],
            columns=["Pred. 0 (normal)", "Pred. 1 (tensión)"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"Verdaderos negativos: {cm[0, 0]} · Falsos positivos: {cm[0, 1]} · "
            f"Falsos negativos: {cm[1, 0]} · Verdaderos positivos: {cm[1, 1]}"
        )

    with col_roc:
        st.markdown(f"**Curva ROC** (AUC = {m['roc_auc']:.3f})")
        roc_df = pd.DataFrame(
            {
                "Tasa de falsos positivos": result["fpr"],
                "Modelo": result["tpr"],
                "Referencia (azar)": result["fpr"],
            }
        ).set_index("Tasa de falsos positivos")
        st.line_chart(roc_df)

    with st.expander("Reporte de clasificación completo"):
        st.text(classification_report(result["y_test"], result["y_pred"], digits=3))

with tab_importancia:
    st.subheader("Importancia de variables (feature_importances_)")
    imp_df = result["importance_df"]
    st.bar_chart(imp_df.set_index("variable")["importancia"])
    st.dataframe(imp_df, use_container_width=True)
