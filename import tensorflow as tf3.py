from numpy._core.multiarray import MAY_SHARE_EXACT
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#Datos
#encabezado
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
column_names = ["Temp", "Price", "VH"]
total_data = pd.read_excel("VHRNA.xlsx", header=0, names=column_names)

X = total_data.drop("VH", axis = 1)
y = total_data["VH"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

#se entrena el modelo
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)

model = Sequential()
model.add(Dense(12, input_shape = (2,), activation = "relu"))
model.add(Dense(8, activation = "relu"))
model.add(Dense(6, activation = "relu"))
model.add(Dense(1, activation = "linear")) # Changed activation to 'linear' for regression

model.compile(loss = "mean_squared_error", optimizer = "Adam",  metrics =["MAE"]) # Changed loss to 'mean_squared_error' and removed accuracy metric
model

# Ajustar el modelo de keras en el conjunto de datos
historial=model.fit(X_train, y_train, epochs = 150, batch_size = 10)

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

# For regression, accuracy is not a suitable metric. We will evaluate loss instead.
loss = model.evaluate(X_train, y_train)

print(f"Loss on training data: {loss}")

y_pred = model.predict(X_test)
y_pred[:15]

#redondeo de la salida
y_pred_round = [round(x[0]) for x in y_pred]
y_pred_round[:15]

print("Hagamos una pronostico!")
# Convert the list to a NumPy array and reshape for prediction
new_data = np.array([[13.45457378, 1.045420514]])
resultado = model.predict(new_data)
print("The pronostico is: " + str(round(resultado[0][0])))