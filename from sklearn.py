from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Crear un clasificador de árbol de decisión para clasificación
clf_classification = DecisionTreeClassifier()
clf_classification.fit(X, y)

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(clf_classification, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Árbol de Decisión para Clasificación (Iris Dataset)')
plt.show()
