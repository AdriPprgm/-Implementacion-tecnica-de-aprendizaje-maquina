#Implementacion de algoritmo DBSCAN con libreria sklearn
import sklearn
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = sklearn.datasets.load_breast_cancer(as_frame=True).frame

counter = df["target"].value_counts()

print("Distribución de clases:")
print(counter)

df.to_csv("Con_Framework/data/breast_cancer.csv") # Guardamos el dataset en un archivo CSV para su posterior uso.

target = "target" #0 o 1 representando la clase de la muestra (maligno o benigno)
features = df.columns[df.columns != target] #Todas las columnas menos la columna target

#Visualización de los datos originales

plt.figure(figsize=(8, 6))
plt.scatter(
    df[features[0]],
    df[features[1]],
    c=df[target],
    cmap="viridis"
)

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Datos Originales")
plt.colorbar(label="Target")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[features], df['target'], test_size=0.3, random_state=42)

# DBSCAN agrupa datos basandose en cercania y densidad.
# Las caracteristicas de nuestro dataset son las "dimensiones" de los datos.
# Con esas mismas dimensiones es como DBSCAN determina si un punto pertenece a un cluster o es ruido.
# Ya que tenemos un dataset con muchas dimensiones, es matematicamente complejo calcular distancias entre datos con tantas dimensiones. 
# Tener muchas dimensiones ademas causa un problema muy conocido en DBSCAN conocido como el "Curse of Dimensionality". 
# A medida que aumenta el número de dimensiones las distancias entre puntos tienden a volverse más uniformes por lo queremos reducir las dimensiones lo mas posible.
# Para esto, vamos a utilizar PCA (Principal Component Analysis)

# Aplicar PCA para reducir la dimensionalidad
# Antes de decidir cuantas dimensiones queremos, calculamos la varianza explicada por cada componente principal para decidir cuantas dimensiones queremos.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca_full = PCA().fit(X_train_scaled)
print(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()


# Se puede ver en la grafica que el 95% de varianza se explica entre el componente 9 y 10. 
# Entones, con esa informacion, podemos reducir las caracteristicas a 10 y mantener el 95% de la informacion original.

#Preparar los datos para DBSCAN
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

from sklearn.neighbors import NearestNeighbors
# Calcular las distancias a los k vecinos más cercanos
nbrs = NearestNeighbors(n_neighbors=5).fit(X_train_pca)

#Encontrar los k vecinos de un punto
distances, indices = nbrs.kneighbors(X_train_pca)
print(distances[:5])  # Muestra las distancias a los 5 vecinos más cercanos para las primeras 5 muestras
print(indices[:5])  # Muestra los índices de los 5 vecinos más cercanos para las primeras 5 muestras

sorted_distances = np.sort(distances, axis=0)
print(sorted_distances[:5])  # Muestra las distancias ordenadas a los 5 vecinos más cercanos para las primeras 5 muestras

#Plotear la "knee" para identificar los puntos que pertenecen al cluster y los que son ruido
k_distances = sorted_distances[:, 4]  # Tomamos la distancia al 5to vecino más cercano

# Antes de plotear, obtener donde esta el "knee" exacto
import kneed
from kneed import KneeLocator

knee_locator = KneeLocator(x = range(1, len(distances)+1), y = k_distances,  S=1.0, curve="convex", direction="increasing", online=True)

knee_point = knee_locator.knee_y

print(f"Knee point: {knee_point}")
plt.figure(figsize=(8, 6))
plt.plot(k_distances)
plt.axhline(y=knee_point, color='r', linestyle='--', label=f'Knee Point: {knee_point:.2f}')
plt.xlabel("Muestras ordenadas por distancia (4th NN)")
plt.ylabel("Distancia al 5to vecino más cercano")
plt.show()

#EPS sera el knee que encontramos, y min_samples sera 5 (ya que es el numero de vecinos que usamos para calcular la distancia)
clusters_train =  DBSCAN(eps=knee_point, min_samples=5).fit(X_train_pca)

from collections import Counter
print(Counter(clusters_train.labels_))

df_scaled_features = pd.DataFrame(X_train_pca)

import seaborn as sns
p = sns.scatterplot(data=df_scaled_features, x=df_scaled_features[0], y=df_scaled_features[1], hue=clusters_train.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.show()


# Aun con PCA, no pudimos reducir las dimensiones suficiente y los efectos de "Curse of Dimensionality" se siguen viendo, 
# dandonos solo un cluster mal distribuido con muchos datos clasificados como ruido.