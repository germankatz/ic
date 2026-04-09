# Modulo K-medias reutilizable

import numpy as np


def k_means(X, k=3, max_iters=100, tol=1e-4, historial=False):
    """
    Implementacion simple de k-medias.
    Retorna: labels, centroids, inertia
    Si historial=True, retorna ademas una lista de (labels, centroids) por iteracion.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=k, replace=False) # choice(limite, cantidad, si repite)
    centroids = X[indices]
    hist = []

    for _ in range(max_iters):
        # Matriz de distancias: cada fila es un punto, cada columna es un centroide
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):       # recorro cada punto
            for j in range(k):           # recorro cada centroide
                # Calculo la distancia euclidea entre el punto i y el centroide j
                distances[i, j] = np.linalg.norm(X[i] - centroids[j])
        # Para cada punto, me quedo con el indice del centroide mas cercano
        labels = np.argmin(distances, axis=1)

        if historial:
            hist.append((labels.copy(), centroids.copy()))

        # Recalculo cada centroide como el promedio de los puntos asignados a el
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            # Puntos que pertenecen al cluster j
            puntos_del_cluster = X[labels == j]
            if len(puntos_del_cluster) > 0:
                # El nuevo centroide es el promedio de esos puntos
                new_centroids[j] = puntos_del_cluster.mean(axis=0)
            else:
                # Si ningun punto fue asignado, dejo el centroide donde estaba
                new_centroids[j] = centroids[j]

        # Cuanto se movieron los centroides respecto a la iteracion anterior
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            break
        centroids = new_centroids

    # Estado final: recalculo distancias y labels con los centroides definitivos
    distances = np.zeros((n_samples, k))
    for i in range(n_samples):
        for j in range(k):
            distances[i, j] = np.linalg.norm(X[i] - centroids[j])
    labels = np.argmin(distances, axis=1)

    # Inercia: suma de las distancias al cuadrado de cada punto a su centroide
    # Cuanto menor sea, mejor es el agrupamiento
    inertia = np.sum((X - centroids[labels])**2)

    if historial:
        hist.append((labels.copy(), centroids.copy()))
        return labels, centroids, inertia, hist
    return labels, centroids, inertia


def cohesion(X, labels, centroids):
    """Cohesion (SSW): suma de distancias al cuadrado de cada punto a su centroide.
    Cuanto menor, mas compactos son los clusters."""
    k = centroids.shape[0]
    return sum(
        np.sum((X[labels == c] - centroids[c]) ** 2)
        for c in range(k)
    )


def separacion(X, labels, centroids):
    """Separacion (SSB): suma ponderada de distancias al cuadrado entre cada centroide
    y el centroide global. Cuanto mayor, mas separados estan los clusters."""
    k = centroids.shape[0]
    centroide_global = np.mean(X, axis=0)
    return sum(
        np.sum(labels == c) * np.sum((centroids[c] - centroide_global) ** 2)
        for c in range(k)
    )


def davies_bouldin(X, labels, centroids):
    """Davies-Bouldin: promedio de la peor similitud entre cada cluster y los demas.
    Valores bajos indican clusters compactos y bien separados."""
    k = centroids.shape[0]
    # Dispersion promedio de cada cluster (distancia media al centroide)
    s = np.zeros(k)
    for c in range(k):
        puntos = X[labels == c]
        if len(puntos) > 0:
            s[c] = np.mean(np.linalg.norm(puntos - centroids[c], axis=1))

    db = 0.0
    for i in range(k):
        peor = 0.0
        for j in range(k):
            if i != j:
                d_ij = np.linalg.norm(centroids[i] - centroids[j])
                if d_ij > 0:
                    r_ij = (s[i] + s[j]) / d_ij
                    peor = max(peor, r_ij)
        db += peor
    return db / k


def fowlkes_mallows(y_true, y_pred):
    """Fowlkes-Mallows (FM): media geometrica de precision y recall sobre pares.
    Va de 0 a 1, donde 1 = clustering perfecto respecto a las clases reales."""
    n = len(y_true)
    tp, fp, fn = 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            mismo_pred = (y_pred[i] == y_pred[j])
            mismo_true = (y_true[i] == y_true[j])
            if mismo_pred and mismo_true:
                tp += 1
            elif mismo_pred and not mismo_true:
                fp += 1
            elif not mismo_pred and mismo_true:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return np.sqrt(precision * recall)


def rand_index(y_true, y_pred):
    """Rand Index ajustado: fraccion de pares correctamente agrupados/separados,
    ajustado por azar. Va de -1 a 1, donde 1 = acuerdo perfecto."""
    n = len(y_true)
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            mismo_pred = (y_pred[i] == y_pred[j])
            mismo_true = (y_true[i] == y_true[j])
            if mismo_pred and mismo_true:
                tp += 1
            elif not mismo_pred and not mismo_true:
                tn += 1
            elif mismo_pred and not mismo_true:
                fp += 1
            else:
                fn += 1
    # RI sin ajustar
    total = tp + tn + fp + fn
    ri = (tp + tn) / total if total > 0 else 0

    # Ajuste por azar
    from math import comb
    n_pares = comb(n, 2)
    # Tabla de contingencia
    clases = np.unique(y_true)
    clusters = np.unique(y_pred)
    nij = np.zeros((len(clases), len(clusters)), dtype=int)
    for idx_c, c in enumerate(clases):
        for idx_k, k in enumerate(clusters):
            nij[idx_c, idx_k] = np.sum((y_true == c) & (y_pred == k))
    a = np.array([np.sum(y_true == c) for c in clases])
    b = np.array([np.sum(y_pred == k) for k in clusters])

    sum_comb_nij = sum(comb(int(nij[i, j]), 2) for i in range(len(clases)) for j in range(len(clusters)))
    sum_comb_a = sum(comb(int(ai), 2) for ai in a)
    sum_comb_b = sum(comb(int(bj), 2) for bj in b)

    esperado = sum_comb_a * sum_comb_b / n_pares if n_pares > 0 else 0
    maximo = (sum_comb_a + sum_comb_b) / 2
    if maximo == esperado:
        return 1.0 if sum_comb_nij == esperado else 0.0
    return (sum_comb_nij - esperado) / (maximo - esperado)
