# Modulo reutilizable: Algoritmo Genético

import numpy as np


class AlgoritmoGenetico:
    def __init__(self, long_cromosoma, funcion_fitness, poblacion_size=30, generaciones=200,
                 prob_cruza=0.8, prob_mutacion=0.01):
        """
        long_cromosoma: cantidad total de bits por individuo
        funcion_fitness: función que recibe un individuo (array de 0/1) y retorna un valor numérico
                         (menor = mejor, se minimiza)
        poblacion_size: cantidad de individuos en la población
        generaciones: número de iteraciones
        prob_cruza: probabilidad de cruzamiento entre dos padres (se evalúa por par de padres)
        prob_mutacion: probabilidad de flip de cada bit (se evalúa por cada bit de cada individuo)
        """
        self.long_cromosoma = long_cromosoma
        self.funcion_fitness = funcion_fitness
        self.poblacion_size = poblacion_size
        self.generaciones = generaciones
        self.prob_cruza = prob_cruza
        self.prob_mutacion = prob_mutacion

    def inicializar_poblacion(self):
        """Crea una población aleatoria de 0s y 1s."""
        poblacion = np.random.randint(0, 2, (self.poblacion_size, self.long_cromosoma))
        return poblacion

    def evaluar(self, poblacion):
        """Evalúa el fitness de cada individuo de la población."""
        valores = np.zeros(self.poblacion_size)
        for i in range(self.poblacion_size):
            valores[i] = self.funcion_fitness(poblacion[i])
        return valores

    def seleccion_torneo(self, poblacion, valores):
        """Selección por competencia (tournament selection de tamaño 2)."""
        padres = []
        for _ in range(self.poblacion_size):
            i1, i2 = np.random.randint(0, self.poblacion_size, 2)
            if valores[i1] < valores[i2]:
                padres.append(poblacion[i1].copy())
            else:
                padres.append(poblacion[i2].copy())
        return np.array(padres)

    def cruza(self, padres):
        """Cruza de un punto entre pares consecutivos de padres.
        Se tira un dado por cada par: si sale < prob_cruza, se cruzan."""
        hijos = []
        for i in range(0, self.poblacion_size, 2): # Recorre de 2 en 2 para tener pares consecutivos
            p1 = padres[i]
            p2 = padres[i + 1]
            if np.random.rand() < self.prob_cruza:
                punto = np.random.randint(1, self.long_cromosoma - 1)
                h1 = np.concatenate([p1[:punto], p2[punto:]])
                h2 = np.concatenate([p2[:punto], p1[punto:]])
            else:
                h1 = p1.copy()
                h2 = p2.copy()
            hijos.append(h1)
            hijos.append(h2)
        return np.array(hijos)

    def mutacion(self, hijos):
        """Mutación: por cada bit de cada individuo se tira un dado,
        si sale < prob_mutacion, ese bit se flippea."""
        for ind in hijos:
            for j in range(self.long_cromosoma):
                if np.random.rand() < self.prob_mutacion:
                    ind[j] = 1 - ind[j]  # flip bit (0->1, 1->0)
        return hijos

    def ejecutar(self):
        """
        Ejecuta el algoritmo genético completo.
        Retorna: (mejor_individuo, mejor_valor)
            mejor_individuo: array de 0/1 con el mejor cromosoma encontrado
            mejor_valor: valor de fitness del mejor individuo
        """
        poblacion = self.inicializar_poblacion()

        mejor_individuo = None
        mejor_valor = np.inf

        for gen in range(self.generaciones):
            # Evaluar fitness
            valores = self.evaluar(poblacion)

            # Guardar el mejor de la generación
            idx_best = np.argmin(valores)
            if valores[idx_best] < mejor_valor:
                mejor_valor = valores[idx_best]
                mejor_individuo = poblacion[idx_best].copy()

            # Selección
            padres = self.seleccion_torneo(poblacion, valores)

            # Cruza
            hijos = self.cruza(padres)

            # Mutación
            hijos = self.mutacion(hijos)

            # Nueva generación
            poblacion = hijos

        return mejor_individuo, mejor_valor


# Utilidades de decodificación

def decode_1var(individuo, dominio, n_bits):
    """
    Decodifica un cromosoma binario a un valor real en el dominio dado.
    individuo: array de 0/1 (longitud n_bits)
    dominio: (xmin, xmax)

    Paso 1: Convertir binario a entero.
        Se recorre bit a bit de izquierda a derecha.
        En cada paso se hace shift a la izquierda (<<1) y OR con el bit actual.
        Ejemplo con [1, 0, 1, 1]:
            paso 1: 0 << 1 | 1 = 1       (en binario: 1)
            paso 2: 1 << 1 | 0 = 2       (en binario: 10)
            paso 3: 2 << 1 | 1 = 5       (en binario: 101)
            paso 4: 5 << 1 | 1 = 11      (en binario: 1011)
        Resultado: 11

    Paso 2: Escalar el entero al dominio real [xmin, xmax].
        El entero va de 0 a (2^n_bits - 1), lo mapeamos linealmente:
        real = xmin + (valor_entero / (2^n_bits - 1)) * (xmax - xmin)
        Con n_bits=4, el entero 11 de 15 posibles y dominio [-10, 10]:
        real = -10 + (11/15) * 20 = 4.67
    """
    # Paso 1: binario a entero
    valor_entero = 0
    for bit in individuo:
        valor_entero = (valor_entero << 1) | bit

    # Paso 2: entero a real en el dominio
    xmin, xmax = dominio
    real = xmin + (valor_entero / (2**n_bits - 1)) * (xmax - xmin)
    return real


def decode_2var(individuo, dominio, n_bits):
    """
    Decodifica un cromosoma binario a dos valores reales (x, y).
    individuo: array de 0/1 (longitud 2*n_bits)
    dominio: (xmin, xmax), mismo para ambas variables

    El cromosoma se parte a la mitad:
        [b0, b1, ..., b(n-1), b(n), b(n+1), ..., b(2n-1)]
         |---- bits de x ----|  |------ bits de y --------|
    Cada mitad se decodifica por separado con decode_1var.
    """
    bits_x = individuo[:n_bits]
    bits_y = individuo[n_bits:]
    x = decode_1var(bits_x, dominio, n_bits)
    y = decode_1var(bits_y, dominio, n_bits)
    return x, y
