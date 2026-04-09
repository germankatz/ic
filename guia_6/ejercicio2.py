import numpy as np
import matplotlib.pyplot as plt
from ag import AlgoritmoGenetico

# Guia 6 - Ejercicio 2
# Selección de características para clasificación de leucemia (ALL vs AML)
# usando Algoritmo Genético + KNN como clasificador

# --- Carga de datos ---

def cargar_datos(ruta):
    """Carga CSV sin encabezado: 7129 features + 1 label (última columna)."""
    datos = np.loadtxt(ruta, delimiter=',')
    X = datos[:, :-1]   # features (n_muestras x 7129)
    y = datos[:, -1]     # labels 0 o 1
    return X, y

X_train, y_train = cargar_datos('guia_6/leukemia_train.csv')
X_test, y_test = cargar_datos('guia_6/leukemia_test.csv')

print(f"Train: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
print(f"Test:  {X_test.shape[0]} muestras, {X_test.shape[1]} features")

# --- Pre-filtrado de features ---
# 7129 bits es demasiado para el AG, así que primero seleccionamos las
# TOP_N features más discriminativas (mayor diferencia de medias entre clases).

TOP_N = 60  # el AG buscará el mejor subconjunto dentro de estos 60

# Diferencia absoluta de medias entre clase 0 y clase 1
media_clase0 = np.mean(X_train[y_train == 0], axis=0)
media_clase1 = np.mean(X_train[y_train == 1], axis=0)
diferencias = np.abs(media_clase0 - media_clase1)

# Índices de las TOP_N features más discriminativas
indices_top = np.argsort(diferencias)[-TOP_N:]  # los TOP_N con mayor diferencia

# Recortamos los datos a esas features
X_train_top = X_train[:, indices_top]
X_test_top = X_test[:, indices_top]

# Normalización min-max al rango [0, 1] (usando stats del train)
mins = X_train_top.min(axis=0)
maxs = X_train_top.max(axis=0)
rangos = maxs - mins
rangos[rangos == 0] = 1  # evitar división por cero

X_train_norm = (X_train_top - mins) / rangos
X_test_norm = (X_test_top - mins) / rangos


# --- Clasificador KNN ---

def knn_clasificar(X_train, y_train, x_test, k=3):
    """Clasifica x_test usando KNN con distancia euclídea."""
    distancias = np.zeros(len(X_train))
    for i in range(len(X_train)):
        distancias[i] = np.linalg.norm(X_train[i] - x_test)

    # Índices de los k vecinos más cercanos
    indices_k = np.argsort(distancias)[:k]

    # Voto mayoritario
    votos = y_train[indices_k]
    cuenta_0 = np.sum(votos == 0)
    cuenta_1 = np.sum(votos == 1)

    if cuenta_1 > cuenta_0:
        return 1
    return 0


def evaluar_knn(X_tr, y_tr, X_te, y_te, k=3):
    """Evalúa accuracy de KNN sobre un conjunto de test."""
    aciertos = 0
    for i in range(len(X_te)):
        pred = knn_clasificar(X_tr, y_tr, X_te[i], k)
        if pred == y_te[i]:
            aciertos += 1
    return aciertos / len(X_te)


# --- Función de fitness para el AG ---
# Cada individuo es un vector binario de TOP_N bits.
# bit=1 -> la feature se usa, bit=0 -> no se usa.
# Fitness = -(accuracy en train) + penalización por cantidad de features
# (queremos MINIMIZAR, así que negamos la accuracy)

ALPHA = 0.005  # peso de la penalización por cantidad de features

def fitness(individuo):
    # Verificar que al menos 1 feature esté seleccionada
    n_seleccionadas = np.sum(individuo)
    if n_seleccionadas == 0:
        return 1.0  # peor fitness posible

    # Máscara de features seleccionadas
    mascara = individuo == 1 # Devuelve un array de booleanos donde True es la feature seleccionada

    X_tr_sel = X_train_norm[:, mascara] # Selecciona las features seleccionadas
    y_tr = y_train

    # Evaluamos accuracy con KNN sobre el mismo train (leave-one-out simplificado)
    aciertos = 0
    for i in range(len(X_tr_sel)):
        # Dejamos fuera la muestra i
        X_loo = np.delete(X_tr_sel, i, axis=0)
        y_loo = np.delete(y_tr, i)
        pred = knn_clasificar(X_loo, y_loo, X_tr_sel[i], k=3)
        if pred == y_tr[i]:
            aciertos += 1

    accuracy = aciertos / len(X_tr_sel)

    # Penalización por usar muchas features (parsimonia)
    penalizacion = ALPHA * (n_seleccionadas / TOP_N)

    # Minimizamos: menor es mejor
    return (1 - accuracy) + penalizacion


# --- Main ---

if __name__ == "__main__":
    print(f"\nPre-filtrado: {TOP_N} features más discriminativas seleccionadas")
    print(f"El AG buscará el mejor subconjunto dentro de estas {TOP_N}\n")

    # Baseline: KNN con TODAS las features pre-filtradas
    acc_todas = evaluar_knn(X_train_norm, y_train, X_test_norm, y_test, k=3)
    print(f"Baseline (todas {TOP_N} features): Accuracy test = {acc_todas*100:.2f}%")

    # Algoritmo Genético
    ag = AlgoritmoGenetico(
        long_cromosoma=TOP_N,
        funcion_fitness=fitness,
        poblacion_size=30,
        generaciones=50,
        prob_cruza=0.8,
        prob_mutacion=0.02
    )
    mejor_individuo, mejor_fitness = ag.ejecutar()

    # Resultados del AG
    mascara = mejor_individuo == 1
    n_sel = np.sum(mascara)
    print(f"\n[AG] Features seleccionadas: {n_sel} de {TOP_N}")
    print(f"[AG] Fitness (menor=mejor): {mejor_fitness:.4f}")

    # Accuracy en test con las features seleccionadas por el AG
    X_tr_sel = X_train_norm[:, mascara]
    X_te_sel = X_test_norm[:, mascara]
    acc_ag = evaluar_knn(X_tr_sel, y_train, X_te_sel, y_test, k=3)
    print(f"[AG] Accuracy en test: {acc_ag*100:.2f}%")

    # Índices originales de las features seleccionadas
    indices_seleccionados = indices_top[mascara]
    print(f"[AG] Índices originales de features: {sorted(indices_seleccionados)}")

    # --- Gráfico comparativo ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Barras de accuracy
    metodos = [f'Todas ({TOP_N})', f'AG ({n_sel})']
    accuracies = [acc_todas * 100, acc_ag * 100]
    axes[0].bar(metodos, accuracies, color=['steelblue', 'indianred'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy en Test: Todas las features vs AG')
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')

    # Gráfico 2: Features seleccionadas (máscara binaria)
    axes[1].bar(range(TOP_N), mejor_individuo, color='indianred', width=1.0)
    axes[1].set_xlabel('Feature (índice pre-filtrado)')
    axes[1].set_ylabel('Seleccionada (1/0)')
    axes[1].set_title(f'Máscara de features del AG ({n_sel} seleccionadas)')

    plt.tight_layout()
    plt.show()
