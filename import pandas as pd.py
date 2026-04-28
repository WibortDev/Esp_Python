import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#Datos
#encabezado
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
total_data = pd.read_csv("DiabetesP3.csv", header=None, names=column_names)

X = total_data.drop("Outcome", axis = 1)
y = total_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

#se entrena el modelo
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)

model = Sequential()
model.add(Dense(12, input_shape = (8,), activation = "relu"))
model.add(Dense(8, activation = "relu"))
model.add(Dense(6, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = "mean_squared_error", optimizer = "Adam", metrics = ["accuracy"])
model

# Ajustar el modelo de keras en el conjunto de datos
historial=model.fit(X_train, y_train, epochs = 150, batch_size = 10)

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

_, accuracy = model.evaluate(X_train, y_train)

print(f"Accuracy: {accuracy}")

y_pred = model.predict(X_test)
y_pred[:15]

#redondeo de la salida
y_pred_round = [round(x[0]) for x in y_pred]
y_pred_round[:15]

print("Hagamos una predicción!")
# Convert the list to a NumPy array and reshape for prediction
new_data = np.array([[-1.1418515161634994,1.8811295904206005,0.9768047509617303,1.4718224750997435,3.7353863483175385,1.435129451314752,-0.7546560869505219,-0.6161106708099943]])
resultado = model.predict(new_data)
print("The prediction is: " + str(round(resultado[0][0])))