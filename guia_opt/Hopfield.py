import numpy as np

class Hopfield:
    def __init__(self):
        self.rng = np.random.default_rng()

    def entrenar(self, x: np.ndarray[int]):
        # pasar de [0,1] a [-1,1]
        x = np.where(x == 0, -1, 1)

        P = x.shape[0]
        N = x.shape[1]

        # Calcular relacion entre cada pixel como productor matricial de x por x transpuesta
        self.w = np.dot(x.T, x) / P
        # Determinar peso 0 entre neurona y si misma
        np.fill_diagonal(self.w, 0)

        self.N = N

    def recuperar(self, x: np.ndarray[int], max_iter: int = 100) -> list[np.ndarray[int]]:
        x = np.where(x == 0, -1, 1)
        # inicializar y(0)
        y = [x.copy()]

        for i in range(1,max_iter):
            y.append(np.sign(np.matmul(self.w, y[i-1])))
            y[i][y[i] == 0] = 1
            if np.array_equal(y[i-1], y[i]):
                break

        return y
