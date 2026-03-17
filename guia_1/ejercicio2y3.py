import csv
import sys
import tty
import termios
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Guia 1 - Ejercicio 2 y 3

# --- Selector de dataset ---
datasets = [
    ("OR",    "guia_1/OR_trn.csv",    "guia_1/OR_tst.csv"),
    ("XOR",   "guia_1/XOR_trn.csv",   "guia_1/XOR_tst.csv"),
    ("OR 50%","guia_1/OR_50_trn.csv", "guia_1/OR_50_tst.csv"),
    ("OR 90%","guia_1/OR_90_trn.csv", "guia_1/OR_90_tst.csv"),
]

def selector(opciones):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    cursor = 0

    def render():
        sys.stdout.write("\033[2J\033[H")  # limpiar pantalla
        sys.stdout.write("Seleccionar dataset:\r\n\r\n")
        for i, (nombre, _, _) in enumerate(opciones):
            if i == cursor:
                sys.stdout.write(f"  \033[1m> {nombre}\033[0m\r\n")
            else:
                sys.stdout.write(f"    {nombre}\r\n")
        sys.stdout.write("\r\n↑↓ para mover, Enter para confirmar")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        while True:
            render()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                sys.stdin.read(1)   # [
                arrow = sys.stdin.read(1)
                if arrow == "A":    # arriba
                    cursor = (cursor - 1) % len(opciones)
                elif arrow == "B":  # abajo
                    cursor = (cursor + 1) % len(opciones)
            elif ch in ("\r", "\n"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    return cursor

opcion = selector(datasets)
nombre_dataset, archivo_trn, archivo_tst = datasets[opcion]
print(f"Dataset seleccionado: {nombre_dataset}\n")

# --- Carga de datos de entrenamiento ---
entradas = []
salidas = []

with open(archivo_trn, mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradas.append([-1, x1, x2])
        salidas.append(yd)

# --- Entrenamiento del Perceptrón ---
np.random.seed(12123)
pesos = np.random.uniform(-0.5, 0.5, 3)

def funcActivacion(result):
    return 1 if result >= 0 else -1

gamma = 0.01
max_epocas = 100

pesos_por_epoca = []
errores_por_epoca = []

for epoca in range(max_epocas):
    errores_epoca = 0
    for i in range(len(entradas)):
        potencial = np.dot(entradas[i], pesos)
        y = funcActivacion(potencial)
        error = salidas[i] - y
        for j in range(len(pesos)):
            pesos[j] += gamma * error * entradas[i][j]
        if error != 0:
            errores_epoca += 1

    pesos_por_epoca.append(pesos.copy())
    errores_por_epoca.append(errores_epoca)
    print(f"Época {epoca+1}: errores = {errores_epoca}")

    if errores_epoca == 0:
        print("Criterio de finalización alcanzado: epoca sin errores.")
        break

# --- Resultados de entrenamiento ---
print("\nPesos modificados:", pesos)

y_e = [funcActivacion(np.dot(entradas[i], pesos)) for i in range(len(entradas))]
aciertos = sum(1 for i in range(len(y_e)) if y_e[i] == salidas[i])

print("\nResultados Entrenamiento:")
print("Cantidad de casos entrenamiento:", len(salidas))
print("Casos exitosos entrenamiento:", aciertos)
print("Pesos finales:", pesos)

# --- Animación de la evolución de la frontera (una recta por época) ---
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(entradas)):
    if salidas[i] == 1:
        ax.scatter(entradas[i][1], entradas[i][2], color="blue", marker="o", label="Clase 1" if i == 0 else "")
    else:
        ax.scatter(entradas[i][1], entradas[i][2], color="red", marker="x", label="Clase -1" if i == 0 else "")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.legend()
ax.grid(True)

x_vals = np.linspace(-2, 2, 100)
linea, = ax.plot([], [], color="black")

def update(frame):
    p = pesos_por_epoca[frame]
    if p[2] != 0:
        y_vals = (p[0] - p[1] * x_vals) / p[2]
        linea.set_data(x_vals, y_vals)
    ax.set_title(f"{nombre_dataset} — Época {frame+1} — Errores: {errores_por_epoca[frame]}")
    return linea,

ani = animation.FuncAnimation(fig, update, frames=len(pesos_por_epoca), interval=500, repeat=False)
plt.show()

# --- Carga de datos de test ---
entradastest = []
salidastest = []

with open(archivo_tst, mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradastest.append([-1, x1, x2])
        salidastest.append(yd)

# --- Evaluación con test ---
ytest = [funcActivacion(np.dot(entradastest[i], pesos)) for i in range(len(entradastest))]
aciertostest = sum(1 for i in range(len(ytest)) if ytest[i] == salidastest[i])

print("\nResultados Test:")
print("Cantidad de casos test:", len(salidastest))
print("Casos exitosos test:", aciertostest)
