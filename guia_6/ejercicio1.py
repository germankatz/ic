import numpy as np
import matplotlib.pyplot as plt
from ag import AlgoritmoGenetico, decode_1var, decode_2var

# Guia 6 - Ejercicio 1
# Comparar Algoritmo Genético vs Gradiente Descendente para minimizar funciones


# =============================================
# Punto 1.i: f(x) = -x * sin(sqrt(|x|))
# con x en [-512, 512]
# =============================================

def f1(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# Derivada analítica de f1(x) para gradiente descendente
def f1_derivada(x):
    if x == 0:  # Evitamos división por cero
        return 0
    return -np.sin(np.sqrt(np.abs(x))) - (x * np.cos(np.sqrt(np.abs(x)))) / (2 * np.sqrt(np.abs(x)))

# Parámetros del AG 1D
N_BITS_1D = 16
DOMINIO_1D = (-512, 512)

def fitness_1d(individuo):
    x = decode_1var(individuo, DOMINIO_1D, N_BITS_1D)
    return f1(x)

# Gradiente Descendente 1D
def gradiente_descendente_1D(lr=0.1, max_iter=1000, dominio=(-512, 512)):
    x = np.random.uniform(dominio[0], dominio[1])
    mejor_x, mejor_valor = x, f1(x)

    for _ in range(max_iter):
        grad = f1_derivada(x)
        x = x - lr * grad
        x = np.clip(x, dominio[0], dominio[1])

        valor = f1(x)
        if valor < mejor_valor:
            mejor_valor = valor
            mejor_x = x

    return mejor_x, mejor_valor


# =============================================
# Punto 1.ii: f(x,y) = (x^2+y^2)^0.25 * (sin^2(50(x^2+y^2)^0.1) + 1)
# con x, y en [-100, 100]
# =============================================

def f2(x, y):
    r2 = x**2 + y**2
    return (r2**0.25) * (np.sin(50 * (r2**0.1))**2 + 1)

# Gradiente numérico en 2D (aproximación por diferencias finitas)
def gradiente_numerico(xy, h=1e-5):
    x, y = xy
    grad_x = (f2(x + h, y) - f2(x - h, y)) / (2 * h)
    grad_y = (f2(x, y + h) - f2(x, y - h)) / (2 * h)
    return np.array([grad_x, grad_y])

# Parámetros del AG 2D
N_BITS_2D = 12
DOMINIO_2D = (-100, 100)

def fitness_2d(individuo):
    x, y = decode_2var(individuo, DOMINIO_2D, N_BITS_2D)
    return f2(x, y)

# Gradiente Descendente 2D
def gradiente_descendente_2D(lr=0.05, max_iter=2000, dominio=(-100, 100)):
    xy = np.random.uniform(dominio[0], dominio[1], 2)
    mejor_xy, mejor_valor = xy.copy(), f2(*xy)

    for _ in range(max_iter):
        grad = gradiente_numerico(xy)
        xy = xy - lr * grad
        xy = np.clip(xy, dominio[0], dominio[1])

        valor = f2(*xy)
        if valor < mejor_valor:
            mejor_valor = valor
            mejor_xy = xy.copy()

    return mejor_xy, mejor_valor


# =============================================
# Main
# =============================================

if __name__ == "__main__":

    # --- Punto 1.i ---
    print("=== Punto 1.i: f(x) = -x * sin(sqrt(|x|)) ===\n")

    ag1 = AlgoritmoGenetico(
        long_cromosoma=N_BITS_1D,
        funcion_fitness=fitness_1d,
        poblacion_size=50,
        generaciones=400,
        prob_cruza=0.8,
        prob_mutacion=0.01
    )
    mejor_ind_1d, mejor_f_ag1 = ag1.ejecutar()
    mejor_x_ag = decode_1var(mejor_ind_1d, DOMINIO_1D, N_BITS_1D)
    print(f"[AG] Mejor solución: x = {mejor_x_ag:.5f}, f(x) = {mejor_f_ag1:.5f}")

    mejor_x_gd, mejor_f_gd1 = gradiente_descendente_1D()
    print(f"[GD] Mejor solución: x = {mejor_x_gd:.5f}, f(x) = {mejor_f_gd1:.5f}")

    # --- Punto 1.ii ---
    print("\n=== Punto 1.ii: f(x,y) = (x²+y²)^0.25 * (sin²(50(x²+y²)^0.1) + 1) ===\n")

    ag2 = AlgoritmoGenetico(
        long_cromosoma=2 * N_BITS_2D,
        funcion_fitness=fitness_2d,
        poblacion_size=30,
        generaciones=300,
        prob_cruza=0.8,
        prob_mutacion=0.01
    )
    mejor_ind_2d, mejor_f_ag2 = ag2.ejecutar()
    mejor_xy_ag = decode_2var(mejor_ind_2d, DOMINIO_2D, N_BITS_2D)
    print(f"[AG] Mejor solución: (x, y) = {mejor_xy_ag}, f(x,y) = {mejor_f_ag2:.5f}")

    mejor_xy_gd, mejor_f_gd2 = gradiente_descendente_2D()
    print(f"[GD] Mejor solución: (x, y) = {mejor_xy_gd}, f(x,y) = {mejor_f_gd2:.5f}")

    # --- Gráficos ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1.i: función 1D
    xs = np.linspace(-512, 512, 1000)
    ys = f1(xs)
    axes[0].plot(xs, ys, label="f(x)")
    axes[0].scatter(mejor_x_ag, mejor_f_ag1, color="red", label="Mínimo AG", zorder=5)
    axes[0].scatter(mejor_x_gd, mejor_f_gd1, color="green", label="Mínimo GD", zorder=5)
    axes[0].set_title("1.i: AG vs GD (1D)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico 1.ii: función 2D
    X = np.linspace(-100, 100, 300)
    Y = np.linspace(-100, 100, 300)
    XX, YY = np.meshgrid(X, Y)
    Z = (XX**2 + YY**2)**0.25 * (np.sin(50 * (XX**2 + YY**2)**0.1)**2 + 1)
    axes[1].contourf(XX, YY, Z, levels=100, cmap="jet")
    axes[1].scatter(mejor_xy_ag[0], mejor_xy_ag[1], color="red", label="Mínimo AG", zorder=5)
    axes[1].scatter(mejor_xy_gd[0], mejor_xy_gd[1], color="green", label="Mínimo GD", zorder=5)
    axes[1].set_title("1.ii: AG vs GD (2D)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
