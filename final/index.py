"""
Examen Final - Inteligencia Computacional
Clasificación de precursores de miRNA en virus

Problema: dado un vector de 28 características, determinar si una secuencia
          de RNA es un precursor de miRNA (clase 1) o no (clase 0).

Dataset:  237 positivos (miRNA) + 839 negativos → fuerte desbalance de clases.

Clasificadores propuestos:
  1. MLP  – Perceptrón Multicapa custom   (final/mlp.py = guia_2/mlp.py)
  2. KNN  – K-Vecinos más Cercanos custom (guia_6/ejercicio2.py)

Validación: StratifiedKFold-5 (extensión del KFold de guia_3 que mantiene
            la proporción de clases en cada fold — necesario por el desbalance).

Métricas: Precisión, Recall y F1 sobre la clase positiva (miRNA).
          NO se usa accuracy porque el desbalance la hace engañosa:
          predecir siempre "no-miRNA" daría 78 % de accuracy pero sería inútil.
          - Recall alto   → detectar la mayor cantidad posible de miRNAs conocidos.
          - Precisión alta → cuando predice miRNA, que sea muy probable que lo sea
                             (confirmar experimentalmente un candidato es muy caro).
          El F1 combina ambas en una sola cifra para facilitar la comparación.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from tabulate import tabulate

from mlp import MLP   # guia_2/mlp.py (mismo código que final/mlp.py)


# =====================================================================
# KNN custom — código extraído de guia_6/ejercicio2.py
# =====================================================================

def knn_clasificar(X_train, y_train, x_test, k=3):
    """Clasifica x_test usando KNN con distancia euclídea."""
    distancias = np.zeros(len(X_train))
    for i in range(len(X_train)):
        distancias[i] = np.linalg.norm(X_train[i] - x_test)
    indices_k = np.argsort(distancias)[:k]
    votos = y_train[indices_k]
    cuenta_0 = np.sum(votos == 0)
    cuenta_1 = np.sum(votos == 1)
    if cuenta_1 > cuenta_0:
        return 1
    return 0


def knn_predecir(X_tr, y_tr, X_te, k=3):
    """Devuelve un array de predicciones para cada fila de X_te."""
    preds = np.zeros(len(X_te))
    for i in range(len(X_te)):
        preds[i] = knn_clasificar(X_tr, y_tr, X_te[i], k)
    return preds


# ==================== Carga de datos ====================
datos = np.loadtxt('final/virus.csv', delimiter=',')
X = datos[:, :-1]   # 28 características
y = datos[:, -1]    # etiqueta: 1 = miRNA, 0 = no-miRNA

print("=" * 57)
print("DATASET")
print("=" * 57)
print(f"Total muestras:            {len(y)}")
print(f"Clase positiva (miRNA):    {int(np.sum(y == 1))}")
print(f"Clase negativa (no-miRNA): {int(np.sum(y == 0))}")
print(f"Dimensión de cada muestra: {X.shape[1]} características")

# ==================== División train / test ====================
# 80 % entrenamiento, 20 % prueba.
# stratify=y mantiene la proporción de clases en ambos conjuntos.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalización: StandardScaler ajustado SOLO sobre train (patrón de guia_3)
scaler = StandardScaler()
X_train_full_n = scaler.fit_transform(X_train_full)
X_test_n       = scaler.transform(X_test)

print(f"\nTrain: {len(y_train_full)} muestras | Test: {len(y_test)} muestras")

# El MLP de guia_2 usa sigmoide simétrica con salida en [-1, 1].
# Convertimos las etiquetas {0, 1} → {-1, 1} para que sean compatibles.
# En predicción usamos: salida > 0  →  clase 1 (miRNA)
#                       salida ≤ 0  →  clase 0 (no-miRNA)
y_train_bp = np.where(y_train_full == 1, 1.0, -1.0)
y_test_bp  = np.where(y_test == 1,       1.0, -1.0)


# ==================== StratifiedKFold ====================
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


# =====================================================================
# CLASIFICADOR 1: MLP custom (final/mlp.py = guia_2/mlp.py)
#
# Arquitectura: capas densas con sigmoide simétrica y retropropagación.
# Paramétrico: aprende una función no lineal ajustando los pesos w.
# =====================================================================

def cv_mlp(estructura, tasa, max_ep, X_tr, y_tr, skf):
    """
    Validación cruzada del MLP custom.
    Retorna arrays de precisión, recall y F1 por fold.
    Patrón de guia_3/ejercicio1.py aplicado al MLP de guia_2/mlp.py.
    """
    precs, recs, f1s = [], [], []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr,  y_f_tr  = X_tr[tr_idx],  y_tr[tr_idx]
        X_f_val, y_f_val = X_tr[val_idx], y_tr[val_idx]

        # Normalizar dentro del fold (evita data leakage)
        sc = StandardScaler()
        X_f_tr  = sc.fit_transform(X_f_tr)
        X_f_val = sc.transform(X_f_val)

        # Etiquetas bipolar para el MLP
        y_bp = np.where(y_f_tr == 1, 1.0, -1.0)

        # Entrenar; se silencian los prints de época durante la CV
        np.random.seed(42)
        red = MLP(estructura=estructura, tasa_aprendizaje=tasa, max_epocas=max_ep)
        with redirect_stdout(StringIO()):
            red.train(X_f_tr, y_bp, umbral_error=0.01)

        # Predecir: salida > 0 → clase 1, salida ≤ 0 → clase 0
        y_pred = np.array([1 if red.forward(x)[0] > 0 else 0 for x in X_f_val])

        precs.append(precision_score(y_f_val, y_pred, pos_label=1, zero_division=0))
        recs.append(recall_score(y_f_val,     y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_f_val,          y_pred, pos_label=1, zero_division=0))

    return np.array(precs), np.array(recs), np.array(f1s)


print("\n" + "=" * 57)
print("BÚSQUEDA DE HIPERPARÁMETROS: MLP custom (guia_2/mlp.py)")
print("=" * 57)

# Exploración de distintas arquitecturas (número de capas y neuronas)
configs_mlp = [
    ([28, 10,     1], 0.01, 80),
    ([28, 20,     1], 0.01, 80),
    ([28, 30,     1], 0.01, 80),
    ([28, 20, 10, 1], 0.01, 80),
]

tabla_mlp = [["Estructura", "lr", "Precisión", "Recall", "F1", "F1 std"]]
resultados_mlp = []

for (estructura, tasa, ep) in configs_mlp:
    print(f"  {estructura}...", end=' ', flush=True)
    prec, rec, f1 = cv_mlp(estructura, tasa, ep, X_train_full, y_train_full, skf)
    resultados_mlp.append((estructura, tasa, ep, f1.mean()))
    tabla_mlp.append([str(estructura), str(tasa),
                      f"{prec.mean():.4f}", f"{rec.mean():.4f}",
                      f"{f1.mean():.4f}", f"{f1.std():.4f}"])
    print(f"F1={f1.mean():.4f}")

print(tabulate(tabla_mlp, tablefmt="fancy_grid", stralign="center"))
mejor_mlp = max(resultados_mlp, key=lambda x: x[3])
print(f"\nMejor MLP: {mejor_mlp[0]}  (F1 CV = {mejor_mlp[3]:.4f})")


# =====================================================================
# CLASIFICADOR 2: KNN custom (guia_6/ejercicio2.py)
#
# Arquitectura: no paramétrico, sin entrenamiento explícito.
# Clasifica cada muestra por voto mayoritario entre sus k vecinos
# más cercanos en el espacio de características (distancia euclídea).
# Arquitectura completamente diferente al MLP.
# =====================================================================

def cv_knn(k, X_tr, y_tr, skf):
    """
    Validación cruzada del KNN custom.
    Patrón de guia_3/ejercicio2.py con knn_clasificar de guia_6/ejercicio2.py.
    """
    precs, recs, f1s = [], [], []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr,  y_f_tr  = X_tr[tr_idx],  y_tr[tr_idx]
        X_f_val, y_f_val = X_tr[val_idx], y_tr[val_idx]

        sc = StandardScaler()
        X_f_tr  = sc.fit_transform(X_f_tr)
        X_f_val = sc.transform(X_f_val)

        y_pred = knn_predecir(X_f_tr, y_f_tr, X_f_val, k)

        precs.append(precision_score(y_f_val, y_pred, pos_label=1, zero_division=0))
        recs.append(recall_score(y_f_val,     y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_f_val,          y_pred, pos_label=1, zero_division=0))

    return np.array(precs), np.array(recs), np.array(f1s)


print("\n" + "=" * 57)
print("BÚSQUEDA DE HIPERPARÁMETROS: KNN custom (guia_6/ejercicio2.py)")
print("=" * 57)

ks = [1, 3, 5, 7, 9, 11]

tabla_knn = [["k", "Precisión", "Recall", "F1", "F1 std"]]
resultados_knn = []

for k in ks:
    print(f"  k={k}...", end=' ', flush=True)
    prec, rec, f1 = cv_knn(k, X_train_full, y_train_full, skf)
    resultados_knn.append((k, f1.mean()))
    tabla_knn.append([str(k), f"{prec.mean():.4f}", f"{rec.mean():.4f}",
                      f"{f1.mean():.4f}", f"{f1.std():.4f}"])
    print(f"F1={f1.mean():.4f}")

print(tabulate(tabla_knn, tablefmt="fancy_grid", stralign="center"))
mejor_k = max(resultados_knn, key=lambda x: x[1])[0]
print(f"\nMejor KNN: k={mejor_k}")


# ==================== Evaluación final sobre test ====================
# Hiperparámetros ya fijados → entrenamos sobre TODO el train y evaluamos
# sobre el test (datos nunca vistos durante la búsqueda).
print("\n" + "=" * 57)
print("EVALUACIÓN FINAL SOBRE TEST")
print("=" * 57)

# --- MLP final ---
mejor_estr, mejor_tasa, mejor_ep, _ = mejor_mlp
np.random.seed(42)
red_final = MLP(estructura=mejor_estr, tasa_aprendizaje=mejor_tasa, max_epocas=mejor_ep)
with redirect_stdout(StringIO()):
    red_final.train(X_train_full_n, y_train_bp, umbral_error=0.01)

y_pred_mlp = np.array([1 if red_final.forward(x)[0] > 0 else 0 for x in X_test_n])

prec_mlp = precision_score(y_test, y_pred_mlp, pos_label=1)
rec_mlp  = recall_score(y_test,    y_pred_mlp, pos_label=1)
f1_mlp   = f1_score(y_test,        y_pred_mlp, pos_label=1)

# --- KNN final ---
y_pred_knn = knn_predecir(X_train_full_n, y_train_full, X_test_n, k=mejor_k)

prec_knn = precision_score(y_test, y_pred_knn, pos_label=1)
rec_knn  = recall_score(y_test,    y_pred_knn, pos_label=1)
f1_knn   = f1_score(y_test,        y_pred_knn, pos_label=1)

tabla_final = [
    ["Clasificador",        "Precisión", "Recall", "F1"],
    [f"MLP {mejor_estr}", f"{prec_mlp:.4f}", f"{rec_mlp:.4f}", f"{f1_mlp:.4f}"],
    [f"KNN k={mejor_k}",  f"{prec_knn:.4f}", f"{rec_knn:.4f}", f"{f1_knn:.4f}"],
]
print(tabulate(tabla_final, tablefmt="fancy_grid", stralign="center"))

ganador = f"MLP {mejor_estr}" if f1_mlp >= f1_knn else f"KNN k={mejor_k}"
print(f"\nClasificador seleccionado: {ganador}")
print("Criterio: mayor F1-score sobre el conjunto de test.")

# ==================== Gráficos ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
ConfusionMatrixDisplay(cm_mlp, display_labels=['No-miRNA', 'miRNA']).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(
    f"MLP {mejor_estr}\n"
    f"Prec={prec_mlp:.2f}  Rec={rec_mlp:.2f}  F1={f1_mlp:.2f}"
)

cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=['No-miRNA', 'miRNA']).plot(
    ax=axes[1], colorbar=False, cmap='Oranges')
axes[1].set_title(
    f"KNN k={mejor_k}\n"
    f"Prec={prec_knn:.2f}  Rec={rec_knn:.2f}  F1={f1_knn:.2f}"
)

plt.suptitle("Matrices de confusión – Clase positiva: miRNA", fontsize=13)
plt.tight_layout()
plt.savefig('final/resultados.png', dpi=150)
plt.show()
