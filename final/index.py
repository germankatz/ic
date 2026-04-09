"""
Examen Final - Inteligencia Computacional
Clasificación de precursores de miRNA en virus

Problema: Dado un vector de 28 características, determinar si una secuencia de RNA
          es un precursor de miRNA (clase 1) o no (clase 0).

Dataset:  237 positivos (miRNA) + 839 negativos → fuerte desbalance de clases.

Clasificadores propuestos:
  1. MLP  – Perceptrón Multicapa (sklearn MLPClassifier, patrón de guia_3/ejercicio1.py)
  2. KNN  – K-Vecinos más Cercanos (sklearn KNeighborsClassifier, patrón de guia_3/ejercicio2.py)

Validación: StratifiedKFold con 5 folds (extiende el patrón de guia_3 con KFold,
            pero estratificado para respetar la proporción de clases en cada fold).

Métricas: Precisión, Recall y F1-score sobre la clase positiva (miRNA).
          NO se usa accuracy porque el dataset está desbalanceado: un clasificador
          que siempre predice "no-miRNA" tendría ~78 % de accuracy pero sería inútil.
          - Recall alto   → detectar la mayor cantidad posible de miRNAs conocidos.
          - Precisión alta → cuando predice miRNA, que sea muy probable que lo sea
                             (experimento biológico de confirmación es muy caro).
          El F1 combina ambas en una sola cifra, facilitando la comparación.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from tabulate import tabulate

# ==================== Carga de datos ====================
datos = np.loadtxt('final/virus.csv', delimiter=',')
X = datos[:, :-1]   # 28 características
y = datos[:, -1]    # etiqueta: 1 = miRNA, 0 = no-miRNA

print("=" * 55)
print("DATASET")
print("=" * 55)
print(f"Total muestras:            {len(y)}")
print(f"Clase positiva (miRNA):    {int(np.sum(y == 1))}")
print(f"Clase negativa (no-miRNA): {int(np.sum(y == 0))}")
print(f"Dimensión de cada muestra: {X.shape[1]} características")

# ==================== Normalización ====================
# StandardScaler: media 0, desviación estándar 1.
# Se ajusta SOLO sobre el conjunto de entrenamiento para evitar data leakage.
# Patrón idéntico al de guia_3/ejercicio1.py y ejercicio2.py.
scaler = StandardScaler()

# ==================== División train / test ====================
# 80 % entrenamiento, 20 % prueba, estratificada para mantener
# la proporción de clases en ambos conjuntos.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalizar con stats del train (aplicar la misma transformación al test)
X_train_full_n = scaler.fit_transform(X_train_full)
X_test_n       = scaler.transform(X_test)

print(f"\nTrain: {len(y_train_full)} muestras | Test: {len(y_test)} muestras")

# ==================== Validación cruzada estratificada ====================
# StratifiedKFold mantiene la proporción de clases en cada fold.
# Con datos desbalanceados, un KFold simple podría generar folds con muy
# pocos ejemplos positivos, sesgando la estimación del desempeño.
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def evaluar_cv(clf_fn, X_tr, y_tr, skf):
    """
    Ejecuta StratifiedKFold sobre X_tr/y_tr instanciando clf_fn() en cada fold.
    Retorna arrays de precisión, recall y F1 por fold.
    Patrón de guia_3/ejercicio1.py y ejercicio2.py.
    """
    precisiones, recalls, f1s = [], [], []
    for train_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr, X_f_val = X_tr[train_idx], X_tr[val_idx]
        y_f_tr, y_f_val = y_tr[train_idx], y_tr[val_idx]

        # Normalizar dentro del fold para no filtrar info del val al train
        sc = StandardScaler()
        X_f_tr  = sc.fit_transform(X_f_tr)
        X_f_val = sc.transform(X_f_val)

        clf = clf_fn()
        clf.fit(X_f_tr, y_f_tr)
        y_pred = clf.predict(X_f_val)

        precisiones.append(precision_score(y_f_val, y_pred, pos_label=1, zero_division=0))
        recalls.append(recall_score(y_f_val, y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_f_val, y_pred, pos_label=1, zero_division=0))

    return np.array(precisiones), np.array(recalls), np.array(f1s)


# =====================================================================
# CLASIFICADOR 1: MLP (Perceptrón Multicapa)
# Arquitectura con capas densas completamente conectadas y activación
# logística (sigmoide). Aprende representaciones no lineales de los datos.
# Patrón: guia_3/ejercicio1.py y ejercicio2.py (MLPClassifier de sklearn).
# =====================================================================
print("\n" + "=" * 55)
print("BÚSQUEDA DE HIPERPARÁMETROS: MLP")
print("=" * 55)

# Configuraciones de capas ocultas a explorar
configs_mlp = [
    (10,),
    (20, 10),
    (20, 10, 5),
    (30, 15),
    (50, 25),
]

tabla_mlp = [["Capas ocultas", "Precisión", "Recall", "F1 (media)", "F1 (std)"]]
resultados_mlp = []

for config in configs_mlp:
    def make_mlp(cfg=config):
        return MLPClassifier(
            hidden_layer_sizes=cfg,
            learning_rate_init=0.005,
            max_iter=500,
            activation='logistic',
            early_stopping=True,
            validation_fraction=0.2,
            shuffle=True,
            random_state=0
        )

    prec, rec, f1 = evaluar_cv(make_mlp, X_train_full, y_train_full, skf)
    resultados_mlp.append((config, f1.mean(), f1.std()))
    tabla_mlp.append([
        str(config),
        f"{prec.mean():.4f}",
        f"{rec.mean():.4f}",
        f"{f1.mean():.4f}",
        f"{f1.std():.4f}"
    ])

print(tabulate(tabla_mlp, tablefmt="fancy_grid", stralign="center"))
mejor_config_mlp = max(resultados_mlp, key=lambda x: x[1])[0]
print(f"\nMejor configuración MLP (mayor F1): {mejor_config_mlp}")


# =====================================================================
# CLASIFICADOR 2: KNN (K-Vecinos más Cercanos)
# Clasificador no paramétrico: asigna la clase mayoritaria entre los k
# vecinos más cercanos en el espacio de características.
# Arquitectura completamente diferente al MLP: no aprende pesos, opera
# directamente sobre los datos de entrenamiento en tiempo de inferencia.
# Patrón: guia_3/ejercicio2.py (KNeighborsClassifier de sklearn).
# =====================================================================
print("\n" + "=" * 55)
print("BÚSQUEDA DE HIPERPARÁMETROS: KNN")
print("=" * 55)

ks = [1, 3, 5, 7, 9, 11, 15]

tabla_knn = [["k", "Precisión", "Recall", "F1 (media)", "F1 (std)"]]
resultados_knn = []

for k in ks:
    def make_knn(k_=k):
        return KNeighborsClassifier(n_neighbors=k_)

    prec, rec, f1 = evaluar_cv(make_knn, X_train_full, y_train_full, skf)
    resultados_knn.append((k, f1.mean(), f1.std()))
    tabla_knn.append([
        str(k),
        f"{prec.mean():.4f}",
        f"{rec.mean():.4f}",
        f"{f1.mean():.4f}",
        f"{f1.std():.4f}"
    ])

print(tabulate(tabla_knn, tablefmt="fancy_grid", stralign="center"))
mejor_k = max(resultados_knn, key=lambda x: x[1])[0]
print(f"\nMejor k para KNN (mayor F1): {mejor_k}")


# ==================== Evaluación final sobre test ====================
# Con los hiperparámetros ya fijados, entrenamos sobre TODO el train
# y evaluamos sobre el test (datos nunca vistos durante la búsqueda).
print("\n" + "=" * 55)
print("EVALUACIÓN FINAL SOBRE TEST")
print("=" * 55)

# --- MLP final ---
mlp_final = MLPClassifier(
    hidden_layer_sizes=mejor_config_mlp,
    learning_rate_init=0.005,
    max_iter=500,
    activation='logistic',
    early_stopping=True,
    validation_fraction=0.2,
    shuffle=True,
    random_state=0
)
mlp_final.fit(X_train_full_n, y_train_full)
y_pred_mlp = mlp_final.predict(X_test_n)

prec_mlp = precision_score(y_test, y_pred_mlp, pos_label=1)
rec_mlp  = recall_score(y_test, y_pred_mlp, pos_label=1)
f1_mlp   = f1_score(y_test, y_pred_mlp, pos_label=1)

# --- KNN final ---
knn_final = KNeighborsClassifier(n_neighbors=mejor_k)
knn_final.fit(X_train_full_n, y_train_full)
y_pred_knn = knn_final.predict(X_test_n)

prec_knn = precision_score(y_test, y_pred_knn, pos_label=1)
rec_knn  = recall_score(y_test, y_pred_knn, pos_label=1)
f1_knn   = f1_score(y_test, y_pred_knn, pos_label=1)

tabla_final = [
    ["Clasificador", "Precisión", "Recall", "F1"],
    [f"MLP {mejor_config_mlp}", f"{prec_mlp:.4f}", f"{rec_mlp:.4f}", f"{f1_mlp:.4f}"],
    [f"KNN k={mejor_k}",        f"{prec_knn:.4f}", f"{rec_knn:.4f}", f"{f1_knn:.4f}"],
]
print(tabulate(tabla_final, tablefmt="fancy_grid", stralign="center"))

# Selección del mejor clasificador según F1
if f1_mlp >= f1_knn:
    ganador = f"MLP con capas {mejor_config_mlp}"
else:
    ganador = f"KNN con k={mejor_k}"
print(f"\nClasificador seleccionado: {ganador}")
print("Criterio: mayor F1-score sobre el conjunto de test.")

# ==================== Gráficos ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
ConfusionMatrixDisplay(cm_mlp, display_labels=['No-miRNA', 'miRNA']).plot(
    ax=axes[0], colorbar=False, cmap='Blues'
)
axes[0].set_title(
    f"MLP {mejor_config_mlp}\n"
    f"Prec={prec_mlp:.2f}  Rec={rec_mlp:.2f}  F1={f1_mlp:.2f}"
)

cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=['No-miRNA', 'miRNA']).plot(
    ax=axes[1], colorbar=False, cmap='Oranges'
)
axes[1].set_title(
    f"KNN k={mejor_k}\n"
    f"Prec={prec_knn:.2f}  Rec={rec_knn:.2f}  F1={f1_knn:.2f}"
)

plt.suptitle("Matrices de confusión – Clase positiva: miRNA", fontsize=13)
plt.tight_layout()
plt.savefig('final/resultados.png', dpi=150)
plt.show()
