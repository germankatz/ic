# Ejercicio 1: SOM bidimensional y unidimensional con circulo.csv y te.csv
# Graficar mapa topologico con lineas entre neuronas vecinas
# Colorear datos segun neurona ganadora y agregar centroides
# Repetir con SOM 1D para te.csv

import numpy as np
import matplotlib.pyplot as plt
import csv
from som import matriz_vectores_2d, matriz_vectores_1d, actualizar_pesos, dibujar_som_2d, dibujar_som_1d

plt.ion()  # Modo interactivo para animacion en tiempo real

# Funciones de tasa de aprendizaje y radio de vecindad
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

# ==================== Carga de datos ====================

def cargar_csv(archivo):
    datos = []
    with open(archivo, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        for fila in lector:
            x1, x2 = map(float, fila)
            datos.append([x1, x2])
    return np.array(datos)

# ==================== SOM 2D con circulo.csv (animado) ====================
print("========== SOM 2D - circulo.csv ==========")

circulo = cargar_csv('guia_4/circulo.csv')
epochs = 1000

matriz_circulo = matriz_vectores_2d(5, 5, 2)

fig1, ax1 = plt.subplots(num=1, figsize=(7, 7))
dibujar_som_2d(ax1, matriz_circulo, circulo, titulo="SOM 2D circulo - Inicio")
fig1.canvas.draw()
fig1.canvas.flush_events()
plt.pause(0.5)

for epoch in range(epochs):
    radio_vecindad = vecindad_fun(epoch)
    tasa_de_aprendizaje = tasa_aprendizaje_fun(epoch)
    for xi in np.random.permutation(circulo):
        matriz_circulo = actualizar_pesos(matriz_circulo, xi, tasa_de_aprendizaje, radio_vecindad)
    if epoch % 10 == 0 or epoch == epochs - 1:
        dibujar_som_2d(ax1, matriz_circulo, circulo, titulo=f"SOM 2D circulo - Epoca {epoch}")
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        plt.pause(0.005)

plt.pause(1)

# ==================== SOM 2D con te.csv (animado) ====================
print("\n========== SOM 2D - te.csv ==========")

te = cargar_csv('guia_4/te.csv')

matriz_te = matriz_vectores_2d(5, 5, 2)

fig2, ax2 = plt.subplots(num=2, figsize=(7, 7))
dibujar_som_2d(ax2, matriz_te, te, titulo="SOM 2D te - Inicio")
fig2.canvas.draw()
fig2.canvas.flush_events()
plt.pause(0.5)

for epoch in range(epochs):
    radio_vecindad = vecindad_fun(epoch)
    tasa_de_aprendizaje = tasa_aprendizaje_fun(epoch)
    for xi in np.random.permutation(te):
        matriz_te = actualizar_pesos(matriz_te, xi, tasa_de_aprendizaje, radio_vecindad)
    if epoch % 10 == 0 or epoch == epochs - 1:
        dibujar_som_2d(ax2, matriz_te, te, titulo=f"SOM 2D te - Epoca {epoch}")
        fig2.canvas.draw()
        fig2.canvas.flush_events()
        plt.pause(0.01)

plt.pause(1)

# ==================== SOM 1D con te.csv (animado, misma cantidad de neuronas: 25) ====================
print("\n========== SOM 1D - te.csv ==========")

n_neuronas_1d = 25  # misma cantidad que el SOM 2D (5x5)
matriz_1d = matriz_vectores_1d(n_neuronas_1d, 2)

fig3, ax3 = plt.subplots(num=3, figsize=(7, 7))
dibujar_som_1d(ax3, matriz_1d, te, titulo="SOM 1D te - Inicio")
fig3.canvas.draw()
fig3.canvas.flush_events()
plt.pause(0.5)

for epoch in range(epochs):
    eta = tasa_aprendizaje_fun(epoch)
    rad = vecindad_fun(epoch, radio_max=n_neuronas_1d // 2)
    for xi in np.random.permutation(te):
        matriz_1d = actualizar_pesos(matriz_1d, xi, eta, rad)
    if epoch % 10 == 0 or epoch == epochs - 1:
        dibujar_som_1d(ax3, matriz_1d, te, titulo=f"SOM 1D te - Epoca {epoch}")
        fig3.canvas.draw()
        fig3.canvas.flush_events()
        plt.pause(0.01)

plt.ioff()
plt.show()
