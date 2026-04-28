import tensorflow as tf
import numpy as np
X = np.array([[0,0,0],
             [0,0,1],
             [0,1,1],
             [1,0,0],
             [1,1,0],
             [1,1,1]], dtype=int)
Y = np.array([0,1,3,4,6,7], dtype=int)
#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[3])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
print("Comenzando entrenamiento...")
historial = modelo.fit(X, Y, epochs=100, verbose=False)
print("Modelo entrenado!")
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.savefig("resultado2.png")
plt.close()

print("Hagamos una predicción!")
# Convert the list to a NumPy array
resultado = modelo.predict(np.array([[1,0,1]]))
print("El resultado es " + str(resultado) + "!")