import numpy as np
import matplotlib.pyplot as plt
from pso import PSO

# Ejercicio 1 - Guía 7: Enjambre de Partículas (PSO)

# f1(x): función 1D
def f1(x):
    """f1(x) = -x * sin(sqrt(|x|))"""
    x = np.array(x)
    return -x * np.sin(np.sqrt(np.abs(x)))

# f2(x,y): funcion 2D
def f2(xy):
    """f2(x,y) = (x² + y²)^0.25 * [sin²(50*(x²+y²)^0.1) + 1]"""
    x, y = xy
    r2 = x**2 + y**2
    return (r2**0.25) * (np.sin(50*(r2**0.1))**2 + 1)


# Main
if __name__ == "__main__":

    # === PSO 1D ===
    print("=== PSO 1D ===")
    pso1d = PSO(
        dimension=1,
        funcion_fitness=f1,
        dominio=(-512, 512),
        num_particulas=60,
        iteraciones=100,
        vmax_fraction=0.5
    )
    x_best, val_best, hist, t, ev, pos1D = pso1d.ejecutar()
    print(f"Mejor x = {x_best:.4f}, f(x) = {val_best:.4f}, tiempo = {t:.3f}s, evals = {ev}")

    # Graficar función 1D y partículas
    xs = np.linspace(-512, 512, 2000)
    ys = f1(xs)

    plt.figure()
    plt.plot(xs, ys, 'k-', label='f1(x)')

    for it, pos in enumerate(pos1D):
        plt.scatter(pos[:, 0], f1(pos[:, 0]), s=15, alpha=0.6,
                    label="Partículas" if it == 0 else "")

    plt.title("PSO 1D: función y posiciones de partículas")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # === PSO 2D ===
    print("\n=== PSO 2D ===")
    pso2d = PSO(
        dimension=2,
        funcion_fitness=f2,
        dominio=(-100, 100),
        num_particulas=100,
        iteraciones=60,
        vmax_fraction=0.2
    )
    xy_best, val_best, hist2D, t2, ev2, pos2D = pso2d.ejecutar()
    print(f"Mejor xy = {xy_best}, f(x,y) = {val_best:.4f}, tiempo = {t2:.3f}s, evals = {ev2}")

    # Graficar función 2D y partículas
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    Xg, Yg = np.meshgrid(x, y)
    Z = np.array([[f2((xx, yy)) for xx, yy in zip(Xrow, Yrow)] for Xrow, Yrow in zip(Xg, Yg)])

    plt.figure()
    plt.contourf(Xg, Yg, Z, levels=50, cmap='viridis')

    for it, pos in enumerate(pos2D[::5]):
        plt.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.6, color='white')

    plt.title("PSO 2D: posiciones de partículas sobre f2(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="f2(x,y)")
    plt.show()


    # Graficar convergencia 1D vs 2D
    plt.plot(hist, label="PSO 1D")
    plt.plot(hist2D, label="PSO 2D")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor f(x)")
    plt.legend()
    plt.title("Convergencia PSO")
    plt.grid(True)
    plt.show()
