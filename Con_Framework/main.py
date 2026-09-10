# Este es el archivo principal que ejecuta el framework de clustering para el dataset principal, en este caso siendo 
# el data set Land Mines de UCI Machine Learning Repository.
import kneed
from kneed import KneeLocator
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Con_Framework/data/landmines.csv")
target = "Mine"
features = [col for col in df.columns if col != target]

# Analisis exploratorio de datos
print(df.head())
print()
print("Distribución de clases (Mine):")
print(df['Mine'].value_counts())
print()
print("Valores únicos de Soil_type (tipo de suelo):")
print(df['Soil_type'].value_counts())
print()
print(df.dtypes)
print()
print(df.describe())

# Visualización de los datos originales (2D)
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
plt.colorbar(label="Tipo de Mina (Mine)")
plt.savefig("Con_Framework/Imagenes_resultados/datos_originales_2D.png", dpi=150)
plt.show()

# Visualización de los datos originales (3D)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['Voltage'],
    df['Height_sensor'],
    df['Soil_type'],
    c=df['Mine'],
    cmap='viridis',
    s=40
)

ax.set_xlabel('Voltage')
ax.set_ylabel('Height_sensor')
ax.set_zlabel('Soil_type')
ax.set_title('Datos Originales - Land Mines (3D, espacio completo de features)')

fig.colorbar(scatter, label='Tipo de mina', shrink=0.6)
plt.savefig("Con_Framework/Imagenes_resultados/datos_originales_3D.png", dpi=150)
plt.show()

# Preparación de los datos para el clustering
# Aunque se va a evaluar un modelo no supervisado, despues de investigar un poco, encontre que se recomendaba hacer un split train/test.
# El split train/test permite validar si los hiperparámetros de DBSCAN (eps, min_samples), 
# seleccionados mediante búsqueda sobre el conjunto de entrenamiento, generalizan a datos no vistos. 
# Como DBSCAN no cuenta con un método .predict() nativo, 
# los clusters obtenidos en entrenamiento se propagan al conjunto de prueba asignando cada punto nuevo al cluster de su punto núcleo (core point) más cercano (mediante 1-NN), 
# permitiendo así evaluar el desempeño del modelo de forma honesta sobre datos independientes.
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import NearestNeighbors
# Calcular las distancias a los k vecinos más cercanos
# Para encontrar la mejor cantidad de vecinos y el mejor valor de epsilon, 
# Se le pidio a Claude hacer un grid searfch para encontrar el mejor valor de epsilon y min_samples.

def grid_search_dbscan(X_train_scaled, y_train):
    mejor_ari = -1
    mejor_combo = None
    resultados = []
    for k in [3, 5, 7, 10, 15, 20]:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_train_scaled)
        distances, _ = nbrs.kneighbors(X_train_scaled)
        sorted_distances = np.sort(distances, axis=0)
        k_distances = sorted_distances[:, k-1]

        knee_locator = KneeLocator(
            range(1, len(k_distances)+1), k_distances,
            curve="convex", direction="increasing"
        )
        knee_k = knee_locator.knee_y
        if knee_k is None:
            print(f"k={k}: no se encontró knee, se omite")
            continue

        for factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            eps_test = knee_k * factor
            labels = DBSCAN(eps=eps_test, min_samples=k).fit(X_train_scaled).labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            ari = adjusted_rand_score(y_train, labels)
            resultados.append((k, factor, eps_test, n_clusters, ari))
            if ari > mejor_ari:
                mejor_ari = ari
                mejor_combo = (k, factor, eps_test, n_clusters)

    resultados.sort(key=lambda x: -x[4])
    print("Top 10 combinaciones por ARI:")
    for r in resultados[:10]:
        print(f"min_samples={r[0]}, factor={r[1]}, eps={r[2]:.3f}, clusters={r[3]}, ARI={r[4]:.3f}")

    print(f"\nMejor combinación: min_samples={mejor_combo[0]}, eps={mejor_combo[2]:.3f}, clusters={mejor_combo[3]}, ARI={mejor_ari:.3f}")
    return mejor_combo, mejor_ari

resultados = grid_search_dbscan(X_train_scaled, y_train)

# Resultado: Mejor combinación: min_samples=5, eps=0.561, clusters=7, ARI=0.117

eps = resultados[0][2]  # 0.561
min_samples = resultados[0][0]  # 5

from collections import Counter
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters_train = dbscan.fit(X_train_scaled)

# Analisis de los resultados del clustering
print("Distribución de clusters en train:")
print(Counter(clusters_train.labels_))

# Se nota que 64 muestras fueron clasificadas como ruido (cluster -1), y el resto se distribuye en 7 clusters.
# Nota personal: Senti que tener 64 muestras como ruido es muy alto, entonces decidi hacer un analisis mas profundo de los datos.

# Analisando que tan buen trabajo hizo DBSCAN para encontrar estrcturas densas en los datos.
df_scaled_features = pd.DataFrame(X_train_scaled)

import seaborn as sns
p = sns.scatterplot(data=df_scaled_features, x=df_scaled_features[0], y=df_scaled_features[1], hue=clusters_train.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", title='Clusters')
p.figure.savefig("Con_Framework/Imagenes_resultados/distribucion_clusters_train.png", dpi=150)
plt.show()
# Podemos ver en la grafica que DBSCAN logro encontrar clusters densos, pero tambien hay muchos puntos dispersos que fueron clasificados como ruido.
# Ademas, el hecho de que encuentre esta estructura de densidad no significa que los clusters encontrados correspondan a las clases reales, por eso es importante hacer un analisis de desempeño del modelo.

# Analisando el ruido del clustering, que es el cluster -1, y comparando con las clases reales
comparacion = pd.DataFrame({'cluster': clusters_train.labels_, 'clase_real': y_train.values})
print(comparacion[comparacion['cluster'] == -1]['clase_real'].value_counts())

# Se nota que la clase 2 domina el ruido, puede significar que la clase 2 es simplemente la mas numerosa
# O que la clase 2 es la mas dispersa y por eso DBSCAN no logra agruparla bien.

# Haciendo un analisis mas profundo del ruido, para ver si hay alguna clase que es mas propensa a ser ruido que otra.
# Cuántas muestras de cada clase hay en total en train
total_por_clase = y_train.value_counts()
print("Total por clase en train:")
print(total_por_clase)

# Cuántas de esas se fueron a ruido
ruido_por_clase = comparacion[comparacion['cluster'] == -1]['clase_real'].value_counts()

# Porcentaje de cada clase que terminó como ruido
porcentaje_ruido = (ruido_por_clase / total_por_clase * 100).round(1)
print("\n% de cada clase que fue clasificado como ruido:")
print(porcentaje_ruido)

# Se encontro que todas las clases estan casi perfectamente balanceadas, todas tienen entre 43-50 muestras.
# Con esto descartamos la posibilidad de que la clase 2 domine el ruido por ser la mas numerosa.
# Siguiendo el analisis se nota que las clases 1, 3, 4, 5, entre 11.6% y 14.0% de sus muestras fueron clasificadas como ruido,
# mientras que la clase 2 tiene un 80% de sus muestras clasificadas como ruido.
# Con esto se puede concluir que la clase 2 es la mas dispersa y por eso DBSCAN no logra agruparla bien, y termina clasificando muchas de sus muestras como ruido.

#Continuando con el entrenamiento de DBSCAN
from sklearn.neighbors import KNeighborsClassifier

# Extraer los core points de train (los que sí forman parte de un cluster denso)
core_mask = clusters_train.core_sample_indices_
core_points = X_train_scaled[core_mask]
core_labels = clusters_train.labels_[core_mask]

print(f"Número de core points: {len(core_points)}")

# Entrenar un 1-NN sobre los core points para "propagar" los clusters a test
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(core_points, core_labels)

labels_test = knn.predict(X_test_scaled)

print("Distribución de clusters en test:")
print(Counter(labels_test))

plt.figure(figsize=(8, 6))
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=labels_test, cmap="viridis")
plt.title("Distribución de Clusters en Test")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Con_Framework/Imagenes_resultados/distribucion_clusters_test.png", dpi=150)
plt.show()

# El resultado nos demostro que ninguna muestra de test fue clasificada como ruido, lo cual es un buen resultado, ya que significa que el modelo generaliza bien a datos no vistos.

# Mapeado de clusters a clases reales para test
mapping_df = pd.DataFrame({'cluster': clusters_train.labels_, 'clase_real': y_train.values})

# Para cada cluster (incluyendo -1), encontrar la clase real más común
cluster_to_class = mapping_df.groupby('cluster')['clase_real'].agg(lambda x: x.value_counts().idxmax())
print("Mapeo cluster -> clase real:")
print(cluster_to_class)

# Aplicar el mapeo a las predicciones de test
labels_test_mapped = pd.Series(labels_test).map(cluster_to_class)

print("\n¿Hay valores sin mapear?", labels_test_mapped.isna().sum())

# Resultados: Mapeo cluster -> clase real:
# cluster
# -1    2
#  0    1
#  1    3
#  2    4
#  3    1
#  4    1
#  5    1
#  6    4
# Name: clase_real, dtype: int64

# Hacemos nuestra matriz de confusión para ver el desempeño del modelo en test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, labels_test_mapped, labels=sorted(y_test.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión - Land Mines (DBSCAN)")
plt.savefig("Con_Framework/Imagenes_resultados/confusion_matrix_dbscan.png", dpi=150)
plt.show()

from sklearn.metrics import normalized_mutual_info_score, accuracy_score, classification_report

# Se le pide a Claude que haga un analisis de las metricas de desempeño del modelo en test, incluyendo ARI, NMI, Accuracy y un reporte de clasificación.
print("=== Métricas en TEST ===")
print(f"ARI (clusters originales vs. clase real): {adjusted_rand_score(y_test, labels_test):.3f}")
print(f"NMI (clusters originales vs. clase real): {normalized_mutual_info_score(y_test, labels_test):.3f}")
print(f"Accuracy (tras mapeo cluster->clase): {accuracy_score(y_test, labels_test_mapped):.3f}")
print()
print("Reporte de clasificación (tras mapeo):")
print(classification_report(y_test, labels_test_mapped, zero_division=0))

# Resumiendo los resultados, el modelo DBSCAN termina teniendo un desempeño pobre,
# con un accuracy de 0.196. Es muy probable que esto se deba a que la clase 2 es muy dispersa y DBSCAN no logra agruparla bien, clasificando muchas de sus muestras como ruido.
# Y tambien que 3 features no son suficientes para separar bien las clases, y que el dataset es muy pequeño, con solo 338 muestras.
# Antes de cerrar, el modelo aun con desempeño pobre, si logra identificar minas en general. Entonces como cierre
# se modificara el problema para que sea un problema binario, colapsaremos las 5 clases de minas a dos clases:
# mina (1) y no mina (0). Esto se hace para confirmar que el modelo logra identificar minas en general, aunque no logre identificar el tipo de mina.

print(df['Mine'].unique())

# La documentación de UCI ML Repository indica el orden de las clases como:
# Null, Anti-Tank, Anti-personnel, Booby Trapped Anti-personnel, M14 Anti-personnel
# 1 es Null, 2 es Anti-Tank, 3 es Anti-personnel, 4 es Booby Trapped Anti-personnel, 5 es M14 Anti-personnel
# Ahora crearemos nuestras etiquetas binarias

y_train_binary = (y_train != 1).astype(int)  # 0 = Null, 1 = Cualquier mina
y_test_binary = (y_test != 1).astype(int)

print("Distribución train (binario):")
print(y_train_binary.value_counts())
print("\nDistribución test (binario):")
print(y_test_binary.value_counts())

# Repetimos el grid search para encontrar los mejores hiperparámetros de DBSCAN para el problema binario
resultados_binario = grid_search_dbscan(X_train_scaled, y_train_binary)

# El resultado termino llendo en la direccion contraria
# Nota personal: Todo este analisis con los 3 diferentes datasets que probe me han hecho llegar a la conclusion de que DBSCAN es un algoritmo que no es 
# muy bueno para datasets con muchas dimensiones o alta complejidad en la estructura o distribución, y que es muy sensible a los hiperparámetros eps y min_samples.
# Como paso final, utilizare el dataset de make_moons para probar DBSCAN en un dataset mas simple y ver si logra un mejor desempeño.

from sklearn.datasets import make_moons

# Generar un dataset de make_moons con 1500 muestras y ruido de 10%
features_moons, target_moons = make_moons(n_samples=1500, noise=0.1, random_state=42)

# Visualización del dataset de make_moons
plt.figure(figsize=(8, 6))
plt.scatter(
    features_moons[:, 0],
    features_moons[:, 1],
    c=target_moons,
    cmap="viridis"
)
plt.title("Dataset de make_moons")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Con_Framework/Imagenes_resultados/dataset_make_moons.png", dpi=150)
plt.show()

# Escalando las características del dataset de make_moons
scaler_moons = StandardScaler()
features_moons_scaled = scaler_moons.fit_transform(features_moons)

# Como el dataset de make_moons es simple y tiene solo 2 dimensiones, 
# podemos encontrar los mejores hiperparámetros de DBSCAN de manera simple.
# Calculamos el knee point para determinar un buen valor de epsilon
nbrs_moons = NearestNeighbors(n_neighbors=5).fit(features_moons_scaled)
distances_moons, indices_moons = nbrs_moons.kneighbors(features_moons_scaled)
sorted_distances_moons = np.sort(distances_moons, axis=0)
k_distances_moons = sorted_distances_moons[:, 4]

knee_locator_moons = KneeLocator(
    range(1, len(k_distances_moons)+1), k_distances_moons,
    curve="convex", direction="increasing"
)

knee_point_moons = knee_locator_moons.knee_y
print(f"Knee point para make_moons: {knee_point_moons:.3f}")

cluster_moons = DBSCAN(eps=knee_point_moons, min_samples=5).fit(features_moons_scaled)

df_scaled_features_moons = pd.DataFrame(features_moons_scaled)
moons_plot = sns.scatterplot(data=df_scaled_features_moons, x=df_scaled_features_moons[0], y=df_scaled_features_moons[1], hue=cluster_moons.labels_, legend="full", palette="deep")
sns.move_legend(moons_plot, "upper right", title='Clusters')
moons_plot.figure.savefig("Con_Framework/Imagenes_resultados/distribucion_clusters_make_moons.png", dpi=150)
plt.show()

# Analisis de desempeño del modelo en make_moons
ari_moons = adjusted_rand_score(target_moons, cluster_moons.labels_)
print(f"Adjusted Rand Index para make_moons: {ari_moons:.3f}")

# Resultado: Adjusted Rand Index para make_moons: 0.967

# Con esto se confirma que DBSCAN es un algoritmo que funciona muy bien en datasets simples y con estructuras densas, pero no es adecuado para datasets complejos o con alta dispersión.