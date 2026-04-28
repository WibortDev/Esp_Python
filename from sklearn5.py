import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', context='notebook')

# Se lee el dataset - Importante, cambiar la ruta según la ubicación del archivo.

df_car = pd.read_csv('E:/Users/Will/Descargas/taller/Laboratorio_dataset_car.csv', sep=';')
columns_names = ['Buying','Maintenance','Doors','Person','lug_boot','safety','class']
df_car.columns = columns_names

# Lee las 10 primeras filas del dataset
df_car.head(10)

# Visualizacion la tupla de filas y columnas de dataset
df_car.shape

# Revisa a estructura y tipos de datos
df_car.info()

# Verificamos valores nulos
df_car.isnull().sum()

# Tratamiento de datos duplicados - elimina duplicados
df_car.drop_duplicates(inplace=True)

# Verificamos su aun hay datos nulos
df_car.isnull().sum()

# Funcion conteo y proporcion de datos por columna
def dist(df,target):
    count= df[target].value_counts(normalize=False)
    prop = df[target].value_counts(normalize=True)
    dist = pd.DataFrame({'Freq[N]':count,'Prop[%]': (prop*100).round(2)})
    return dist

# Ver el conteo y la proporción por cada variable (incluida class)
for i in columns_names:
    print(' '*7,i.upper())
    print(dist(df_car,i))
    print("*"*23)

# Graficamos la variable class separada por los distintos atributos
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

for i, variable in enumerate(columns_names[:-1]):
    row = i // 3
    col = i % 3
    sns.countplot(data=df_car, x='class',hue=variable, ax=axes[row][col])
    axes[row][col].set_title(f"Evaluation classes by {variable} Category")

plt.tight_layout()
plt.show()

# Separamos datos por X - Y
X_car = df_car.drop('class',axis=1)
y_car = df_car['class']

# Se realiza el Split
from sklearn.model_selection import train_test_split

X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car,y_car,test_size=0.3,stratify=y_car,random_state=42)

print('X:',X_train_car.shape, X_test_car.shape)
print('y:',y_train_car.shape, y_test_car.shape)

# Se realiza el undersampling
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(random_state=42)
X_train_car, y_train_car = undersample.fit_resample(X_train_car, y_train_car)

# Se realiza elEncoding

import category_encoders as ce 
encoder = ce.OrdinalEncoder(cols=X_train_car.columns)
X_train_car = encoder.fit_transform(X_train_car)
X_test_car = encoder.transform(X_test_car)
X_train_car.head()
X_train_car.dtypes

# Instancias del modelo

from sklearn.tree import DecisionTreeClassifier
tree_car = DecisionTreeClassifier(random_state=42)

#--------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rf_car = RandomForestClassifier(random_state=42,n_jobs=-1)

from sklearn.model_selection import GridSearchCV

# parametros del decision tree
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5]}

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search = GridSearchCV(tree_car, param_grid=param_grid, cv=10, return_train_score=True)
grid_search.fit(X_train_car, y_train_car)

#----------------------------------------------------------------

# parametros del random forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(rf_car,param_grid=param_grid_rf,cv=5,scoring='accuracy',n_jobs=-1)

grid_rf.fit(X_train_car, y_train_car)


# Imprimir los resultados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

print("Mejor puntuación de validación cruzada:")
print(grid_search.best_score_)

print("Mejores hiperparámetros Random Forest:")
print(grid_rf.best_params_)

print("Mejor score de validación cruzada (RF):")
print(grid_rf.best_score_)

# Modelo decision tree con parametros optimizados
best_tree_car = grid_search.best_estimator_

# Predecimos Y
y_train_pred_tree_car = best_tree_car.predict(X_train_car)
y_test_pred_tree_car = best_tree_car.predict(X_test_car)

best_rf_car = grid_rf.best_estimator_

# Predicciones
y_train_pred_rf = best_rf_car.predict(X_train_car)
y_test_pred_rf  = best_rf_car.predict(X_test_car)

# Graficamos matriz de confusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Arbol de decisión
cm = confusion_matrix(y_test_car,y_test_pred_tree_car,labels=best_tree_car.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_tree_car.classes_).plot()

# Bosque aleatorio
cm_rf = confusion_matrix(y_test_car,y_test_pred_rf,labels=best_rf_car.classes_)
ConfusionMatrixDisplay(cm_rf,display_labels=best_rf_car.classes_).plot()
plt.show()

from sklearn.metrics import accuracy_score, classification_report

# Calculo el accuract en train 
train_acc = accuracy_score(y_true=y_train_car,y_pred=y_train_pred_tree_car)
# Calculo el accuract en test 
test_acc  = accuracy_score(y_true=y_test_car,y_pred=y_test_pred_tree_car)

print("Accuracy Decision Tree en train es:",train_acc)
print("Accuracy Decision Tree en test es:",test_acc)

print(classification_report(y_test_car,y_test_pred_tree_car))

# Calculo el accuract en train 
train_acc_rf = accuracy_score(y_train_car, y_train_pred_rf)
# Calculo el accuract en test
test_acc_rf  = accuracy_score(y_test_car,  y_test_pred_rf)

print("Accuracy Random Forest en train:", train_acc_rf)
print("Accuracy Random Forest en test :", test_acc_rf)

print(classification_report(y_test_car, y_test_pred_rf))

# Grafica de Importancia de variables del árbol

feature_scores_car = pd.DataFrame(pd.Series(best_tree_car.feature_importances_, index=X_train_car.columns).sort_values(ascending=False)).T
plt.figure(figsize=(12,6))
sns.barplot(data=feature_scores_car)

for index, value in enumerate(feature_scores_car.values.flatten()):
    plt.annotate(f'{value:.2f}', xy=(index, value), ha='center', va='bottom')

plt.title("Factores clave en la predicción de la calidad de un automovil")
plt.show()
pd.DataFrame(feature_scores_car.T)

# Grafica de Importancia de variables del Bosque
importances_rf = pd.Series(best_rf_car.feature_importances_,index=X_train_car.columns).sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=importances_rf.index,y=importances_rf.values)
plt.xticks(rotation=45, ha='right')
plt.title("Importancia de variables – Random Forest")
plt.xlabel("Variables")
plt.ylabel("Importancia")

for i, v in enumerate(importances_rf.values):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Grafica el árbol
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
tree.plot_tree(best_tree_car,filled=True)
plt.title("Árbol de decisión")
plt.show()

# Grafica del bosque
from sklearn import tree
import matplotlib.pyplot as plt

# Seleccionamos un árbol individual del bosque
tree_in_forest = best_rf_car.estimators_[0]

plt.figure(figsize=(15,10))
tree.plot_tree(tree_in_forest,feature_names=X_train_car.columns,class_names=best_rf_car.classes_,filled=True,max_depth=3)
plt.title("Árbol individual del Random Forest")
plt.show()

results = pd.DataFrame({
    'Modelo': ['Árbol de Decisión', 'Random Forest'],
    'Accuracy Train': [train_acc, train_acc_rf],
    'Accuracy Test':  [test_acc,  test_acc_rf]
})

results