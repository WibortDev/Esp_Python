#Importar librerías necesarias
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
#Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
#Crear y entrenar el árbol
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
#Hacer predicciones
y_pred = modelo.predict(X_test)
#Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
#Visualizar el árbol de decisión
plt.figure(figsize=(25, 15))
plot_tree(modelo, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
#Pronóstico
y_Pron = modelo.predict(X_test)
W=[4.7,3.2,1.6,0.2]
W_Test=modelo.predict([W])
print(W_Test)