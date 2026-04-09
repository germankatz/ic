# Modulo reutilizable: SOM (1D y 2D) y K-medias

import numpy as np
import matplotlib.pyplot as plt


def matriz_vectores_2d(filas, columnas, dimension):
    """Crea una matriz de pesos SOM 2D con valores aleatorios entre -0.5 y 0.5"""
    return np.random.uniform(-0.5, 0.5, (filas, columnas, dimension))


def matriz_vectores_1d(num_neuronas, dimension):
    """Crea una matriz de pesos SOM 1D con valores aleatorios entre -0.5 y 0.5"""
    return np.random.uniform(-0.5, 0.5, (num_neuronas, dimension))


def actualizar_pesos(matriz, x, tasa_de_aprendizaje, radio_vecindad):
    """
    Actualiza pesos del SOM (1D o 2D) con excitacion uniforme en la vecindad.
    matriz: SOM 1D (num_neuronas, dim) o 2D (filas, columnas, dim)
    x: vector de entrada
    tasa_de_aprendizaje: eta
    radio_vecindad: r
    """
    if matriz.ndim == 2:
        # SOM 1D
        distancias = np.linalg.norm(matriz - x, axis=1)
        ganadora = np.argmin(distancias)
        for idx in range(max(0, ganadora - radio_vecindad), min(matriz.shape[0], ganadora + radio_vecindad + 1)):
            matriz[idx] += tasa_de_aprendizaje * (x - matriz[idx])

    elif matriz.ndim == 3:
        # SOM 2D
        filas, columnas = matriz.shape[:2]
        distancia_min = -1
        ganadora = (0, 0)
        for i in range(filas):
            for j in range(columnas):
                d = np.linalg.norm(matriz[i, j] - x)
                if distancia_min == -1 or d < distancia_min:
                    distancia_min = d
                    ganadora = (i, j)

        i_ganadora, j_ganadora = ganadora
        for p in range(max(0, i_ganadora - radio_vecindad), min(filas, i_ganadora + radio_vecindad + 1)):
            for q in range(max(0, j_ganadora - radio_vecindad), min(columnas, j_ganadora + radio_vecindad + 1)):
                matriz[p, q] += tasa_de_aprendizaje * (x - matriz[p, q])
    else:
        raise ValueError("La matriz debe ser 1D o 2D")

    return matriz


def bmu_2d(matriz, x):
    """Retorna la posicion (fila, columna) de la neurona ganadora para x en un SOM 2D."""
    distancias = np.linalg.norm(matriz - x, axis=2)
    return np.unravel_index(np.argmin(distancias), distancias.shape)


def dibujar_som_2d(ax, matriz, datos, titulo="SOM 2D", matriz_full=None, datos_full=None):
    """Dibuja en el ax dado: datos coloreados por BMU + centroides + lineas entre vecinas.
    Si se pasan matriz_full y datos_full, las BMU se calculan en ese espacio
    (ej. 4D) y solo se usa matriz/datos para dibujar la proyeccion 2D."""
    ax.cla()
    filas, columnas = matriz.shape[:2]
    num_neuronas = filas * columnas
    colores = plt.colormaps.get_cmap('tab20').resampled(num_neuronas)

    # Calcular BMU en el espacio completo si se proporciona, sino en el proyectado
    m_bmu = matriz_full if matriz_full is not None else matriz
    d_bmu = datos_full if datos_full is not None else datos

    indices_ganadora = np.zeros(len(datos), dtype=int)
    for idx, x in enumerate(d_bmu):
        distancias = np.linalg.norm(m_bmu - x, axis=2)
        ganadora = np.unravel_index(np.argmin(distancias), distancias.shape)
        indices_ganadora[idx] = ganadora[0] * columnas + ganadora[1]

    for i in range(num_neuronas):
        ax.scatter(datos[indices_ganadora == i, 0], datos[indices_ganadora == i, 1],
                   color=colores(i), s=15, alpha=0.6)

    for i in range(filas):
        for j in range(columnas):
            w = matriz[i, j]
            idx = i * columnas + j
            ax.scatter(w[0], w[1], color=colores(idx), marker='o', s=90, edgecolors='k')

    for i in range(filas):
        for j in range(columnas):
            if i + 1 < filas:
                ax.plot([matriz[i, j, 0], matriz[i+1, j, 0]],
                        [matriz[i, j, 1], matriz[i+1, j, 1]], 'k-', linewidth=0.5)
            if j + 1 < columnas:
                ax.plot([matriz[i, j, 0], matriz[i, j+1, 0]],
                        [matriz[i, j, 1], matriz[i, j+1, 1]], 'k-', linewidth=0.5)

    ax.set_title(titulo)
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.grid(True)


def vecindad_fun(epoch, radio_max=2):
    if epoch < 200:
        return radio_max
    elif epoch < 800:
        t = (epoch - 200) / 600.0
        return max(1, round(radio_max * (1 - t) + 1 * t))
    else:
        return 0


def tasa_aprendizaje_fun(epoch):
    if epoch < 200:
        return 0.8
    elif epoch < 800:
        t = (epoch - 200) / 600.0
        return 0.8 * (1 - t) + 0.1 * t
    else:
        return 0.01


def dibujar_som_1d(ax, matriz_1d, datos, titulo="SOM 1D", matriz_full=None, datos_full=None):
    """Dibuja en el ax dado: datos coloreados por BMU + cadena de neuronas 1D.
    Si se pasan matriz_full y datos_full, las BMU se calculan en ese espacio
    y solo se usa matriz_1d/datos para dibujar la proyeccion 2D."""
    ax.cla()
    n = matriz_1d.shape[0]
    cmap = plt.colormaps.get_cmap('tab20').resampled(n)

    m_bmu = matriz_full if matriz_full is not None else matriz_1d
    d_bmu = datos_full if datos_full is not None else datos
    distancias = np.linalg.norm(d_bmu[:, None, :] - m_bmu[None, :, :], axis=2)
    idx_ganadora = np.argmin(distancias, axis=1)

    for k in range(n):
        pts = datos[idx_ganadora == k]
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], color=cmap(k), s=15, alpha=0.85)

    ax.plot(matriz_1d[:, 0], matriz_1d[:, 1], 'k-', lw=0.8, alpha=0.9)

    for k in range(n):
        w = matriz_1d[k]
        ax.scatter(w[0], w[1], color=cmap(k), edgecolors='k', s=90, zorder=3)

    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal')
    ax.grid(True)