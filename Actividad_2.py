import requests
import pandas as pd
import numpy as np

url = "https://www.postdata.gov.co/api/action/datastore/search.json"

params = {
    "resource_id": "34900bf0-a1a5-4f48-9000-d683869e337e",
    "limit": 5000,

    # Filtro por departamento
    "filters[DESC_DEPARTAMENTO]": "BOGOTÁ D.C.",

    # Columnas requeridas
    "fields[]": [
        "ANNO",
        "MES",
        "DESC_EMPRESA",
        "DESC_DEPARTAMENTO",
        "DESC_MUNICIPIO",
        "DIA",
        "HORA_PICO",
        "PORCENTAJE_INTENTO_NO_EXITOSO"
    ]
}

response = requests.get(url, params=params, timeout=60)
data = response.json()

df = pd.DataFrame(data["result"]["records"])
columnas = [
        "ANNO",
        "MES",
        "DESC_EMPRESA",
        "DESC_DEPARTAMENTO",
        "DESC_MUNICIPIO",
        "DIA",
        "HORA_PICO",
        "PORCENTAJE_INTENTO_NO_EXITOSO"
    ]

df = df[columnas]

# 1) y a numérico (por si viene "12,34" o "12.34")
df["PORCENTAJE_INTENTO_NO_EXITOSO"] = (
    df["PORCENTAJE_INTENTO_NO_EXITOSO"].astype(str)
      .str.replace(",", ".", regex=False)
)
df["PORCENTAJE_INTENTO_NO_EXITOSO"] = pd.to_numeric(df["PORCENTAJE_INTENTO_NO_EXITOSO"], errors="coerce")

# 2) ANNO y MES a numérico (por si vienen como texto)
df["ANNO"] = pd.to_numeric(df["ANNO"], errors="coerce")
df["MES"]  = pd.to_numeric(df["MES"],  errors="coerce")

# Quitar nulos básicos
df = df.dropna(subset=["PORCENTAJE_INTENTO_NO_EXITOSO","ANNO","MES","DIA","HORA_PICO"])

X = df.drop(columns=["PORCENTAJE_INTENTO_NO_EXITOSO", "DESC_DEPARTAMENTO"])
y = df["PORCENTAJE_INTENTO_NO_EXITOSO"]

# Separar columnas numéricas y categóricas
num_cols = ["ANNO", "MES"]
cat_cols = ["DESC_EMPRESA", "DESC_MUNICIPIO", "DIA", "HORA_PICO"]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# OneHotEncoder: genera dummies, handle_unknown evita errores si aparecen categorías nuevas en test
# Para compatibilidad entre versiones:
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

X_train_p = preprocess.fit_transform(X_train)
X_test_p  = preprocess.transform(X_test)

print("Dimensión X_train_p:", X_train_p.shape)

#se entrena el modelo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_p.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

history = model.fit(
    X_train_p, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

train_metrics = model.evaluate(X_train_p, y_train, verbose=0)
test_metrics  = model.evaluate(X_test_p, y_test, verbose=0)

print("Train (loss, mae):", train_metrics)
print("Test  (loss, mae):", test_metrics)

y_pred = model.predict(X_test_p)
print("Predicciones (primeras 10):")
print(y_pred[:10].reshape(-1))

top_real_ym = (
    df.groupby(
        ["ANNO", "MES", "DIA", "HORA_PICO"],
        as_index=False
    )["PORCENTAJE_INTENTO_NO_EXITOSO"]
    .mean()
    .sort_values("PORCENTAJE_INTENTO_NO_EXITOSO", ascending=False)
)

print("Top 10 (Año-Mes-Día-Hora) con mayor % promedio no exitoso:")
print(top_real_ym.head(10))

import matplotlib.pyplot as plt

top10 = top_real_ym.head(10).copy()
top10["dia_hora"] = top10["DIA"].astype(str) + " - " + top10["HORA_PICO"].astype(str)

plt.figure(figsize=(10,5))
plt.barh(top10["dia_hora"][::-1], top10["PORCENTAJE_INTENTO_NO_EXITOSO"][::-1])
plt.xlabel("% promedio intento no exitoso")
plt.ylabel("Día - Hora pico")
plt.title("Top 10 combinaciones Día-Hora con mayor % de intentos no exitosos")
plt.tight_layout()
plt.show()