# Ejercicio 2: K-medias y SOM sobre Iris, matrices de contingencia y graficos
# Comparar soluciones de k-medias y SOM con las clases de referencia
# Graficar datos en 2D coloreados por cluster (k-medias y SOM)
# Graficar neuronas del SOM con frecuencias de activacion e indicar clase

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
from som import matriz_vectores_1d, actualizar_pesos, dibujar_som_1d, vecindad_fun, tasa_aprendizaje_fun
from kmedias import k_means, cohesion, separacion, davies_bouldin, fowlkes_mallows, rand_index

plt.ion()  # Modo interactivo para animacion en tiempo real


# ==================== Carga de datos Iris ====================

x = []
y = []

with open('guia_2/iris81_trn.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    next(lector)  # saltea encabezado
    for fila in lector:
        x1, x2, x3, x4 = map(float, fila[:4])
        x.append([x1, x2, x3, x4])
        y1, y2, y3 = map(float, fila[4:7])
        y.append([y1, y2, y3])

x = np.array(x)
y = np.array(y)
y_true = np.argmax(y, axis=1)
clases = ["Setosa", "Versicolor", "Virginica"]

componente_x = x[:, 0]
componente_y = x[:, 2]

# ==================== K-medias (animado) ====================
print("========== K-medias ==========")

labels_km, centroids_km, inertia, hist_km = k_means(x, k=3, historial=True)
print(f"Inercia k-medias: {inertia:.4f}")

# --- Medidas de calidad de clustering ---
print(f"Cohesion (SSW): {cohesion(x, labels_km, centroids_km):.4f}")
print(f"Separacion (SSB): {separacion(x, labels_km, centroids_km):.4f}")
print(f"Davies-Bouldin: {davies_bouldin(x, labels_km, centroids_km):.4f}")
print(f"Fowlkes-Mallows: {fowlkes_mallows(y_true, labels_km):.4f}")
print(f"Rand Index ajustado: {rand_index(y_true, labels_km):.4f}")

fig_km, ax_km = plt.subplots(num=1, figsize=(8, 6))

for it, (labels_it, centroids_it) in enumerate(hist_km):
    ax_km.cla()
    for cluster in range(3):
        ax_km.scatter(componente_x[labels_it == cluster],
                      componente_y[labels_it == cluster],
                      label=f'Cluster {cluster}', s=15, alpha=0.7)
    ax_km.scatter(centroids_it[:, 0], centroids_it[:, 2],
                  c='black', marker='X', s=200, label='Centroides')
    ax_km.set_xlabel('Sepal length')
    ax_km.set_ylabel('Petal length')
    ax_km.set_title(f'K-medias - Iteracion {it}')
    ax_km.legend()
    ax_km.grid(True)
    fig_km.canvas.draw()
    fig_km.canvas.flush_events()
    plt.pause(0.3)

plt.pause(1)

# ==================== SOM 1D sobre Iris (animado, 3 neuronas) ====================
print("\n========== SOM 1D sobre Iris ==========")

epochs = 1000
n_neuronas = 3
matriz = matriz_vectores_1d(n_neuronas, 4)

datos_proy = x[:, [0, 2]]

fig_som, ax_som = plt.subplots(num=2, figsize=(7, 7))
dibujar_som_1d(ax_som, matriz[:, [0, 2]], datos_proy, titulo="SOM 1D Iris - Inicio",
               matriz_full=matriz, datos_full=x)
fig_som.canvas.draw()
fig_som.canvas.flush_events()
plt.pause(0.5)

for epoch in range(epochs):
    radio_vecindad = vecindad_fun(epoch, radio_max=4)
    tasa_de_aprendizaje = tasa_aprendizaje_fun(epoch)
    for xi in x:
        matriz = actualizar_pesos(matriz, xi, tasa_de_aprendizaje, radio_vecindad)
    if epoch % 10 == 0 or epoch == epochs - 1:
        dibujar_som_1d(ax_som, matriz[:, [0, 2]], datos_proy,
                       titulo=f"SOM 1D Iris - Epoca {epoch}",
                       matriz_full=matriz, datos_full=x)
        fig_som.canvas.draw()
        fig_som.canvas.flush_events()
        plt.pause(0.005)

plt.pause(1)

# ==================== Asignacion de clases a neuronas del SOM ====================

clases_por_neurona = [[] for _ in range(n_neuronas)]
asignacion_clase = np.full(n_neuronas, -1, dtype=int)

for dato, clase_real in zip(x, y_true):
    distancias = np.linalg.norm(matriz - dato, axis=1)
    bmu = np.argmin(distancias)
    clases_por_neurona[bmu].append(clase_real)

for i in range(n_neuronas):
    if clases_por_neurona[i]:
        asignacion_clase[i] = Counter(clases_por_neurona[i]).most_common(1)[0][0]

# Prediccion SOM para cada dato
y_pred_som = []
for dato in x:
    distancias = np.linalg.norm(matriz - dato, axis=1)
    bmu = np.argmin(distancias)
    y_pred_som.append(asignacion_clase[bmu])

y_pred_som = np.array(y_pred_som)

# ==================== Matrices de contingencia ====================
print("\n========== Matrices de contingencia ==========")

# y_true vs SOM
contingencia_true_som = confusion_matrix(y_true, y_pred_som)
df_true_som = pd.DataFrame(contingencia_true_som, index=clases, columns=clases)
print("\ny_true vs SOM:")
print(df_true_som)

# y_true vs K-means
contingencia_true_kmeans = confusion_matrix(y_true, labels_km)
df_true_kmeans = pd.DataFrame(contingencia_true_kmeans, index=clases, columns=clases)
print("\ny_true vs K-means:")
print(df_true_kmeans)

# K-means vs SOM
contingencia_kmeans_som = confusion_matrix(labels_km, y_pred_som)
df_kmeans_som = pd.DataFrame(contingencia_kmeans_som, index=clases, columns=clases)
print("\nK-means vs SOM:")
print(df_kmeans_som)

# ==================== Graficos finales ====================
print("\n========== Graficos finales ==========")

centroides_x = centroids_km[:, 0]
centroides_y = centroids_km[:, 2]

# K-medias final
plt.figure(figsize=(8, 6))
for cluster in range(3):
    plt.scatter(componente_x[labels_km == cluster],
                componente_y[labels_km == cluster],
                label=f'Cluster {cluster}')
plt.scatter(centroides_x, centroides_y, c='black', marker='X', s=200, label='Centroides')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Clustering K-medias (componentes 0 y 2)')
plt.legend()
plt.grid(True)

# SOM final
plt.figure(figsize=(8, 6))
for clase in range(3):
    mask = y_pred_som == clase
    plt.scatter(componente_x[mask], componente_y[mask], label=f'{clases[clase]}')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Clustering SOM (componentes 0 y 2)')
plt.legend()
plt.grid(True)

# ==================== Grafico de neuronas SOM con frecuencias de activacion ====================

frecuencias = np.zeros(n_neuronas)
for dato in x:
    distancias = np.linalg.norm(matriz - dato, axis=1)
    bmu = np.argmin(distancias)
    frecuencias[bmu] += 1

fig, ax = plt.subplots(figsize=(8, 4))
colores_bar = ['tab:blue', 'tab:orange', 'tab:green']
barras = ax.bar(range(n_neuronas), frecuencias, color=[colores_bar[asignacion_clase[i]] if asignacion_clase[i] >= 0 else 'gray' for i in range(n_neuronas)],
                edgecolor='k')

for i in range(n_neuronas):
    clase_txt = clases[asignacion_clase[i]] if asignacion_clase[i] >= 0 else "Sin datos"
    ax.text(i, frecuencias[i] + 1, f"{int(frecuencias[i])}\n{clase_txt}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Neuronas SOM 1D - Frecuencia de activacion y clase asignada')
ax.set_xlabel('Neurona')
ax.set_ylabel('Frecuencia')
ax.set_xticks(range(n_neuronas))
plt.tight_layout()

plt.ioff()
plt.show()
