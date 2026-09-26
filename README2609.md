# Predicción de tensión de liquidez — Random Forest

App de Streamlit que entrena un `RandomForestClassifier` sobre variables de capital de trabajo y predice `Tension_Liquidez_bin` (1 = tensión de liquidez, 0 = situación normal).

## Archivos del proyecto

| Archivo | Función |
|---|---|
| `app.py` | Código de la app: carga de datos, entrenamiento, métricas, gráficas. Punto de entrada para Streamlit. |
| `requirements.txt` | Dependencias de Python que Streamlit Cloud instala antes de correr `app.py`. |
| `Base_Didactica_Random_Forest_Capital_Trabajo.xlsx` | Base didáctica de 600 observaciones simuladas, usada como dato por defecto (hoja `Datos_Modelo`). |

Los tres archivos deben estar en la misma carpeta del repositorio, con esos nombres exactos y en minúsculas.

## Configuración del modelo (aprobada)

- Objetivo: clasificación binaria de `Tension_Liquidez_bin`.
- Predictores: las 21 variables marcadas "Predictor" en el Diccionario de la base (Sector + 20 numéricas/binarias).
- Excluidas del entrenamiento: `ID_Observacion`, `Fecha` (identificador y control), `Prob_Tension_Liquidez` (genera la etiqueta, produce leakage) y `Brecha_Caja_90d_MXN` (target de un modelo de regresión aparte).
- Codificación de `Sector`: One-Hot Encoding dentro de un `ColumnTransformer`.
- División de datos: 70% entrenamiento / 30% prueba, `random_state=42`, estratificada por el target.
- Balance de clases: `class_weight='balanced'` en el Random Forest.
- Hiperparámetros fijos: `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`, `random_state=42`.
- Umbral de clasificación: 0.5 fijo.
- Métricas reportadas: accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, curva ROC, reporte de clasificación completo e importancia de variables (`feature_importances_`).

Resultado de referencia sobre la base didáctica (30% de prueba, 180 observaciones): accuracy 0.850, precision 0.844, recall 0.902, F1 0.872, ROC-AUC 0.924.

## Cómo correr la app en local

```
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`. Por defecto usa la base didáctica incluida; desde la barra lateral se puede cargar un archivo `.xlsx` o `.csv` propio, siempre que conserve las mismas 26 columnas (o al menos las 21 predictoras, `Sector` y el target `Tension_Liquidez_bin`).

## Cómo desplegar en Streamlit Community Cloud

1. Sube los tres archivos a un repositorio de GitHub, en la misma carpeta.
2. En [share.streamlit.io](https://share.streamlit.io), crea la app apuntando a ese repositorio, la rama correspondiente y `app.py` como "Main file path".
3. Streamlit Cloud instala automáticamente `requirements.txt` antes de correr la app. Cualquier cambio a `requirements.txt` requiere "Reboot app" desde "Manage app" para forzar la reinstalación; si el problema persiste, "Delete app" y volver a crearla suele resolver cachés de dependencias corruptas.
4. Si aparece `ModuleNotFoundError`, casi siempre significa que `requirements.txt` no se instaló: revisa en "Manage app" → logs que corra `pip install -r requirements.txt`, y confirma que el archivo está en la misma carpeta que `app.py` y sin extensión oculta (`requirements.txt.txt` en Windows es un error común).

## Estructura de datos esperada

Ver el diccionario completo en la hoja `Diccionario` del archivo `.xlsx`. Resumen de las columnas que la app usa:

- **Predictor categórico:** `Sector` (Manufactura, Comercio, Servicios, Distribución, Tecnología).
- **Predictores numéricos/binarios:** `Ventas_12M_MXN`, `Crecimiento_Ventas_pct`, `Margen_EBITDA_pct`, `DSO_dias`, `DIO_dias`, `DPO_dias`, `CCC_dias`, `CxC_MXN`, `Inventarios_MXN`, `CxP_MXN`, `NWC_MXN`, `Deuda_CP_MXN`, `Caja_MXN`, `Linea_Credito_Disponible_MXN`, `Concentracion_Top5_Clientes_pct`, `Morosidad_CxC_pct`, `Inventario_Obsoleto_pct`, `Volatilidad_Ventas_pct`, `Estacionalidad_bin`, `EBITDA_MXN`.
- **Target:** `Tension_Liquidez_bin`.

## Advertencia sobre los datos

Los datos de `Base_Didactica_Random_Forest_Capital_Trabajo.xlsx` son 100% simulados con fines didácticos y no representan una empresa real (ver hoja `LEEME` del archivo). Las métricas de desempeño reportadas por la app corresponden a esos datos simulados y no son transferibles a un caso de negocio real sin reentrenar con datos propios.
