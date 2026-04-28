from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Crear un regresor de árbol de decisión con un máximo de 4 niveles
clf_regression_small = DecisionTreeRegressor(max_depth=2)
clf_regression_small.fit(X, y)

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(clf_regression_small, feature_names=diabetes.feature_names, filled=True)
plt.title('Árbol de Decisión para Regresión (Diabetes Dataset)')
plt.show()


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
W=[5.8,2.8,4.0,1.2]
W_Test=modelo.predict([W])
print(W_Test)
