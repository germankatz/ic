from math import trunc
import numpy as np
import matplotlib.pyplot as plt
import csv
from ag import AlgoritmoGenetico
from mlp import MLP


from numpy.random import rand

# ==================== Carga de datos ====================

def cargar_csv(archivo):
    datos = np.loadtxt(archivo, delimiter=',')
    X = datos[:, :-1]
    y = datos[:, -1]
    return X, y
    
X, y = cargar_csv('final/virus.csv')

#Datos de la fila 1  
# X[1,:]

# Datos cada fila
# X[:,1]

###################
# Particionar para train y test
# Opcion 1 Reducir todo para que tengan mismo tamaño
###################

# División 80/10/10
# Pero de cada clase
n = len(y)

# 237 clase positiva
# 839 clase negativa

# Datos de entrenamiento
train_y = []
train_x = []

# Datos de prueba
test_y = []
test_x = []

# Solo positivos
for i in range(237):
    if i < trunc(237*0.9):
        # Primeros 213
        train_y.append(y[i])
        train_x.append(X[i + 237])
    else:
        # Resto
        test_y.append(y[i])
        test_x.append(X[i + 237])

# Solo negativos

for i in range(238,237+237):
    if i < trunc(473*0.9):
        # Primeros 213
        train_y.append(y[i])
        train_x.append(X[i + 237])
    else:
        # Resto
        test_y.append(y[i])
        test_x.append(X[i + 237])

# Paso a numpy
train_x = np.array(train_x)
train_y = np.array(train_y)

test_y = np.array(test_y)
test_x = np.array(test_x)

print(train_x.shape)



# ###################
# # Genero nuevos datos ficticios interpolando 
# ###################


# ###################
# # Encontrar parametros más adecuados a ver si hay correlación
# # Debería cambiar no minimizar el accuracy
# ###################


# # --- Pre-filtrado de features ---
# # 7129 bits es demasiado para el AG, así que primero seleccionamos las
# # TOP_N features más discriminativas (mayor diferencia de medias entre clases).

# TOP_N = 15  # el AG buscará el mejor subconjunto dentro de estos 60

# # Diferencia absoluta de medias entre clase 0 y clase 1
# media_clase0 = np.mean(train_x[train_y == 0], axis=0)
# media_clase1 = np.mean(train_x[train_x == 1], axis=0)
# diferencias = np.abs(media_clase0 - media_clase1)

# print("dif")
# print(diferencias)

# # Índices de las TOP_N features más discriminativas
# indices_top = np.argsort(diferencias)[-TOP_N:]  # los TOP_N con mayor diferencia

# # Recortamos los datos a esas features
# X_train_top = train_x[:, indices_top]
# X_test_top = test_x[:, indices_top]

# # Normalización min-max al rango [0, 1] (usando stats del train)
# mins = X_train_top.min(axis=0)
# maxs = X_train_top.max(axis=0)
# rangos = maxs - mins
# rangos[rangos == 0] = 1  # evitar división por cero

# X_train_norm = (X_train_top - mins) / rangos
# X_test_norm = (X_test_top - mins) / rangos


# # --- Clasificador KNN ---

# def knn_clasificar(X_train, y_train, x_test, k=3):
#     """Clasifica x_test usando KNN con distancia euclídea."""
#     distancias = np.zeros(len(X_train))
#     for i in range(len(X_train)):
#         distancias[i] = np.linalg.norm(X_train[i] - x_test)

#     # Índices de los k vecinos más cercanos
#     indices_k = np.argsort(distancias)[:k]

#     # Voto mayoritario
#     votos = y_train[indices_k]
#     cuenta_0 = np.sum(votos == 0)
#     cuenta_1 = np.sum(votos == 1)

#     if cuenta_1 > cuenta_0:
#         return 1
#     return 0


# def evaluar_knn(X_tr, y_tr, X_te, y_te, k=3):
#     """Evalúa accuracy de KNN sobre un conjunto de test."""
#     aciertos = 0
#     for i in range(len(X_te)):
#         pred = knn_clasificar(X_tr, y_tr, X_te[i], k)
#         if pred == y_te[i]:
#             aciertos += 1
#     return aciertos / len(X_te)


# # --- Función de fitness para el AG ---
# # Cada individuo es un vector binario de TOP_N bits.
# # bit=1 -> la feature se usa, bit=0 -> no se usa.
# # Fitness = -(accuracy en train) + penalización por cantidad de features
# # (queremos MINIMIZAR, así que negamos la accuracy)

# ALPHA = 0.005  # peso de la penalización por cantidad de features

# def fitness(individuo):
#     # Verificar que al menos 1 feature esté seleccionada
#     n_seleccionadas = np.sum(individuo)
#     if n_seleccionadas == 0:
#         return 1.0  # peor fitness posible

#     # Máscara de features seleccionadas
#     mascara = individuo == 1 # Devuelve un array de booleanos donde True es la feature seleccionada

#     X_tr_sel = X_train_norm[:, mascara] # Selecciona las features seleccionadas
#     y_tr = train_y

#     # Evaluamos accuracy con KNN sobre el mismo train (leave-one-out simplificado)
#     aciertos = 0
#     for i in range(len(X_tr_sel)):
#         # Dejamos fuera la muestra i
#         X_loo = np.delete(X_tr_sel, i, axis=0)
#         y_loo = np.delete(y_tr, i)
#         pred = knn_clasificar(X_loo, y_loo, X_tr_sel[i], k=3)
#         if pred == y_tr[i]:
#             aciertos += 1

#     accuracy = aciertos / len(X_tr_sel)

#     # Penalización por usar muchas features (parsimonia)
#     penalizacion = ALPHA * (n_seleccionadas / TOP_N)

#     # Minimizamos: menor es mejor
#     return (1 - accuracy) + penalizacion



# print(f"\nPre-filtrado: {TOP_N} features más discriminativas seleccionadas")
# print(f"El AG buscará el mejor subconjunto dentro de estas {TOP_N}\n")

# # Baseline: KNN con TODAS las features pre-filtradas
# acc_todas = evaluar_knn(X_train_norm, train_y, X_test_norm, test_y, k=3)
# print(f"Baseline (todas {TOP_N} features): Accuracy test = {acc_todas*100:.2f}%")

# # Algoritmo Genético
# ag = AlgoritmoGenetico(
#     long_cromosoma=TOP_N,
#     funcion_fitness=fitness,
#     poblacion_size=30,
#     generaciones=5,
#     prob_cruza=0.8,
#     prob_mutacion=0.02
# )
# mejor_individuo, mejor_fitness = ag.ejecutar()

# # Resultados del AG
# mascara = mejor_individuo == 1
# n_sel = np.sum(mascara)
# print(f"\n[AG] Features seleccionadas: {n_sel} de {TOP_N}")
# print(f"[AG] Fitness (menor=mejor): {mejor_fitness:.4f}")

# # Accuracy en test con las features seleccionadas por el AG
# X_tr_sel = X_train_norm[:, mascara]
# X_te_sel = X_test_norm[:, mascara]
# acc_ag = evaluar_knn(X_tr_sel, train_y, X_te_sel, test_y, k=3)
# print(f"[AG] Accuracy en test: {acc_ag*100:.2f}%")

# # Índices originales de las features seleccionadas
# indices_seleccionados = indices_top[mascara]
# print(f"[AG] Índices originales de features: {sorted(indices_seleccionados)}")

# # --- Gráfico comparativo ---
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Gráfico 1: Barras de accuracy
# metodos = [f'Todas ({TOP_N})', f'AG ({n_sel})']
# accuracies = [acc_todas * 100, acc_ag * 100]
# axes[0].bar(metodos, accuracies, color=['steelblue', 'indianred'])
# axes[0].set_ylabel('Accuracy (%)')
# axes[0].set_title('Accuracy en Test: Todas las features vs AG')
# axes[0].set_ylim(0, 105)
# for i, v in enumerate(accuracies):
#     axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')

# # Gráfico 2: Features seleccionadas (máscara binaria)
# axes[1].bar(range(TOP_N), mejor_individuo, color='indianred', width=1.0)
# axes[1].set_xlabel('Feature (índice pre-filtrado)')
# axes[1].set_ylabel('Seleccionada (1/0)')
# axes[1].set_title(f'Máscara de features del AG ({n_sel} seleccionadas)')

# plt.tight_layout()
# plt.show()


######################
# Clasificar
######################

# MLP? 

# Crear red: 2 entradas → 2 ocultas → 1 salida
red = MLP(estructura=[28, 30, 1], tasa_aprendizaje=0.01, max_epocas=100)
red.train(train_x, train_y)

# Prueba
print("prueba")
for x, y in zip(test_y, test_x):
    salida = red.forward(x)
    prediccion = np.round(salida)
    print(f"Entrada: {x}, Esperado: {y}, Predicción: {np.round(salida, 3)}")

# Funcion de base radial