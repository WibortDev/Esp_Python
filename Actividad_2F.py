import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

# ----------------------------
# 1 - DESCARGA DEL DATASET (todas las filas del subconjunto BOGOTÁ D.C.)
# ----------------------------
url = "https://www.postdata.gov.co/api/action/datastore/search.json"

cols = [
    "ANNO","MES","DESC_EMPRESA","DESC_DEPARTAMENTO","DESC_MUNICIPIO",
    "DIA","HORA_PICO","PORCENTAJE_INTENTO_NO_EXITOSO"
]

base_params = {
    "resource_id": "34900bf0-a1a5-4f48-9000-d683869e337e",
    "filters[DESC_DEPARTAMENTO]": "BOGOTÁ D.C.",
    "fields[]": cols
}

# Consultar total
r0 = requests.get(url, params={**base_params, "limit": 1}, timeout=60)
r0.raise_for_status()
total = r0.json()["result"]["total"]
print("Total registros (Bogotá D.C.):", total)

# Descargar en bloques (limit=5000 como venías usando)
limit = 5000
offset = 0
all_records = []

while offset < total:
    r = requests.get(url, params={**base_params, "limit": limit, "offset": offset}, timeout=60)
    r.raise_for_status()
    chunk = r.json()["result"]["records"]
    all_records.extend(chunk)
    offset += limit
    print("Descargados:", min(offset, total))

df = pd.DataFrame(all_records)[cols]
print("Shape final:", df.shape)

# Requisito mínimo: 1000 instancias
if len(df) < 1000:
    raise ValueError(f"El subconjunto descargado tiene {len(df)} instancias (<1000). Ajusta filtro o dataset.")

# ----------------------------
# 2 - CALIDAD LIMPIEZA Y TIPOS
# ----------------------------
# y a numérico (por si viene como string con coma)
df["PORCENTAJE_INTENTO_NO_EXITOSO"] = (
    df["PORCENTAJE_INTENTO_NO_EXITOSO"].astype(str).str.replace(",", ".", regex=False)
)
df["PORCENTAJE_INTENTO_NO_EXITOSO"] = pd.to_numeric(df["PORCENTAJE_INTENTO_NO_EXITOSO"], errors="coerce")

df["ANNO"] = pd.to_numeric(df["ANNO"], errors="coerce")
df["MES"]  = pd.to_numeric(df["MES"], errors="coerce")
df["DIA"]  = pd.to_numeric(df["DIA"], errors="coerce")

df = df.dropna(subset=["PORCENTAJE_INTENTO_NO_EXITOSO","ANNO","MES","DIA","HORA_PICO","DESC_EMPRESA","DESC_MUNICIPIO"])

X = df.drop(columns=["PORCENTAJE_INTENTO_NO_EXITOSO", "DESC_DEPARTAMENTO"])
y = df["PORCENTAJE_INTENTO_NO_EXITOSO"]

# ----------------------------
# 3 - CARACTERIZACIÓN (texto + gráficos)
# ----------------------------
print("\n--- DESCRIPCIÓN (y) ---")
print(y.describe())

print("\n--- NULOS POR COLUMNA ---")
print(df.isna().sum())

# Histograma de la variable objetivo
plt.figure(figsize=(8,4))
plt.hist(y, bins=30)
plt.title("Distribución de % intento no exitoso")
plt.xlabel("% intento no exitoso")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Boxplot de y por hora pico (si HORA_PICO es categórica, muestra dispersión)
plt.figure(figsize=(10,4))
df.boxplot(column="PORCENTAJE_INTENTO_NO_EXITOSO", by="HORA_PICO")
plt.title("Boxplot % no exitoso por HORA_PICO")
plt.suptitle("")
plt.xlabel("HORA_PICO")
plt.ylabel("% intento no exitoso")
plt.tight_layout()
plt.show()

# ----------------------------
# 4 - PREPROCESADO (one-hot + normalización numérica)
# ----------------------------
num_cols = ["ANNO","MES","DIA"]
cat_cols = ["DESC_EMPRESA","DESC_MUNICIPIO","HORA_PICO"]

# Compatibilidad sklearn (sparse_output vs sparse)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", ohe, cat_cols)
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5 - MODELOS NO-NN
# ----------------------------
# 5.1 Regresión lineal
linreg = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LinearRegression())
])

linreg.fit(X_train, y_train)
pred_lr = linreg.predict(X_test)

# 5.2 Random Forest Regressor (parámetros relevantes: n_estimators, max_depth)
rf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# ----------------------------
# 6 - RED NEURONAL (Keras / TF 2.x)
# ----------------------------
X_train_p = preprocess.fit_transform(X_train)
X_test_p  = preprocess.transform(X_test)

set_random_seed(42)

nn = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_p.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

nn.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

history = nn.fit(
    X_train_p, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

pred_nn = nn.predict(X_test_p).reshape(-1)

# Curva de pérdida
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Curvas de pérdida (NN)")
plt.xlabel("Época")
plt.ylabel("MSE loss")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 7 - MÉTRICAS Y COMPARACIÓN GRÁFICA
# ----------------------------
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

m_lr = metrics(y_test, pred_lr)
m_rf = metrics(y_test, pred_rf)
m_nn = metrics(y_test, pred_nn)

results = pd.DataFrame({
    "Modelo": ["LinearRegression", "RandomForest", "NeuralNetwork"],
    "MAE":  [m_lr[0], m_rf[0], m_nn[0]],
    "RMSE": [m_lr[1], m_rf[1], m_nn[1]],
    "R2":   [m_lr[2], m_rf[2], m_nn[2]],
})

print("\n--- RESULTADOS (test) ---")
print(results)

# Comparación superpuesta (barras)
plt.figure(figsize=(9,4))
x = np.arange(len(results["Modelo"]))
plt.bar(x - 0.25, results["MAE"],  width=0.25, label="MAE")
plt.bar(x,        results["RMSE"], width=0.25, label="RMSE")
plt.bar(x + 0.25, results["R2"],   width=0.25, label="R2")
plt.xticks(x, results["Modelo"], rotation=15)
plt.title("Comparación de métricas por modelo (test)")
plt.legend()
plt.tight_layout()
plt.show()

# Predicho vs real (NN)
plt.figure(figsize=(5,5))
plt.scatter(y_test, pred_nn, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("NN: Real vs Predicho (test)")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.tight_layout()
plt.show()

# ----------------------------
# 8 - ANÁLISIS: TOP 10 Año-Mes-Día-Hora
# ----------------------------
top_real_ymdh = (
    df.groupby(["ANNO","MES","DIA","HORA_PICO","DESC_EMPRESA"], as_index=False)["PORCENTAJE_INTENTO_NO_EXITOSO"]
      .mean()
      .sort_values("PORCENTAJE_INTENTO_NO_EXITOSO", ascending=False)
)

print("\nTop 10 (promedio) ANNO-MES-DIA-HORA y PROVEEDOR con mayor % no exitoso:")
print(top_real_ymdh.head(10))
