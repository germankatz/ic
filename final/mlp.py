import numpy as np

# Funcion de activacion sigmoide (simétrica) f(x) = 2 / (1 + exp(-x)) - 1
# Retorna valores en [-1, 1].
def sigmoide(x):
    return 2 / (1 + np.exp(-x)) - 1

# Derivada de la sigmoide simétrica respecto de su salida y: f'(y) = 0.5 * (1+y) * (1-y)
def sigmoide_derivada(y):
    return 0.5 * (1 + y) * (1 - y)


# Clase Capa
class Capa:
    def __init__(self, n_entradas, n_neuronas):
        """
        n_entradas: número de entradas de la capa
        n_neuronas: número de neuronas en esta capa
        self.pesos: matriz de pesos de tamaño (n_neuronas x n_entradas + 1)
                    pesos[i,j] = peso de la conexión de la entrada j hacia la neurona i
        """
        self.pesos = np.random.uniform(-0.5, 0.5, (n_neuronas, n_entradas+1)) # +1 PARA EL BIAS
        self.salida = None    # vector de salida de esta capa (tamaño n_neuronas,)
        self.delta = None     # vector de deltas (gradientes locales)
        self.entradas = None  # vector de entradas recibido (n_entradas)

    def forward(self, entradas):
        entradas = np.hstack(([-1], entradas))   # agregar bias x0 = -1
        self.entradas = entradas
        z = np.dot(self.pesos, entradas)   # z_i = Σ w_ij * x_j
        self.salida = sigmoide(z)
        return self.salida # Salida de la capa, vector (n_neuronas)

    def calcular_delta_salida(self, yd):
        """
        Cálculo del delta en la capa de salida (δ_iii en el apunte):
        δ_i = (y_d - y_i) * f'(y_i)
        -> ver que desde esta capa se propaga el error (y_d - y) hacia atras...
        """
        self.delta = (yd - self.salida) * sigmoide_derivada(self.salida)

    def calcular_delta_oculta(self, pesos_siguiente, delta_siguiente):
        """
        Cálculo del delta en capas ocultas:
        δ_i = f'(y_i) * Σ δ_k * w_ki para cada neurona de la capa siguiente
        donde:
          - pesos_siguiente: matriz (n_neuronas_siguiente x n_neuronas_actual)
          - delta_siguiente: vector (n_neuronas_siguiente,)
        Si estoy en la capa i , necesito los pesos w_(i+1) y los deltas_(i+1)
        """
        self.delta = np.dot(pesos_siguiente[:, 1:].T, delta_siguiente) * sigmoide_derivada(self.salida)
        # pesos_siguiente[:, 1:] para descartar los pesos del bias de la capa siguiente

    def actualizar_pesos(self, tasa_aprendizaje):
        # Actualización de pesos: Δw_ij = ta * δ_i * x_j
        entradas = self.entradas.reshape(1, -1)   # (1 x n_entradas) entradas_traspuesta [-1, x1, x2]
        delta = self.delta.reshape(-1, 1)         # (n_neuronas x 1) [d1,d2,d3]'T
        self.pesos += tasa_aprendizaje * np.dot(delta, entradas)  # (n_neuronas x n_entradas)

        #Aca lo que hace es transponer un vector para luego hacer el producto punto entre delta asi [ d1,d2,d3]^t * [x1,x2,x3]
        #Eso nos va a dar una matriz exactamente de igual dimension que la de los pesos, para asi lograr una suma de matrices de igual dimension


# Clase Perceptrón Multicapa (MLP)
class MLP:
    def __init__(self, estructura, tasa_aprendizaje=0.1, max_epocas=10000):
        """
        estructura: lista con nº de neuronas por capa. Ej: [2, 2, 1]
            -> cantidad de entradas: 2 entradas
            -> primera capa con neuronas: 2 neuronas
            -> cantidad de salidas: 1 neurona
        otro ej: red = MLP(estructura=[4, 5, 4, 2], tasa_aprendizaje=0.1, max_epocas=100)
        que son 4 entradas, primera capa con 5 neuronas, segunda capa con 4 neuronas y capa de salida con 2 neuronas.
        """
        self.capas = []
        self.tasa = tasa_aprendizaje
        self.max_epocas = max_epocas

        for i in range(1, len(estructura)): # Empezamos en 1 porque la capa 0 es la de entradas, no es una capa con neuronas.
            # Creamos cada capa con sus pesos aleatorios
            self.capas.append(Capa(estructura[i-1], estructura[i]))

    def forward(self, entrada):
        # Realiza la propagación hacia adelante de una entrada.
        salida = entrada
        for capa in self.capas:
            salida = capa.forward(salida) # la salida de una capa es la entrada de la siguiente.
        return salida

    def train(self, X, Y, modo='binario', umbral_error=None):
        """
        Entrenamiento con retropropagación, reportando desempeño por época.
        X: matriz de entradas (n_filas x n_entradas)
        Y: matriz de salidas deseadas (n_filas x n_salidas)
        modo: 'binario'    -> clasificación con una salida, usa round para contar aciertos
              'multiclase' -> clasificación con varias salidas, usa argmax para contar aciertos
        umbral_error: si se especifica, detiene el entrenamiento cuando ECM < umbral_error
        """
        history = {"acc": [], "ecm": []}

        for epoca in range(self.max_epocas):
            error_total = 0
            aciertos = 0

            for x, y in zip(X, Y):
                # FORWARD
                salida = self.forward(x) # salida y de la red para la entrada x

                # BACKWARD

                # Delta en capa de salida
                self.capas[-1].calcular_delta_salida(y) # a la ultima capa le calculamos el delta de forma distinta.

                # Delta en capas ocultas (desde la penúltima hasta la primera)
                for i in reversed(range(len(self.capas)-1)):
                    self.capas[i].calcular_delta_oculta(self.capas[i+1].pesos, self.capas[i+1].delta)
                # Acordarse que aca para calcular el delta de una capa i, necesitamos los pesos y deltas de la capa siguiente ii.

                # Actualización de pesos
                for capa in self.capas:
                    capa.actualizar_pesos(self.tasa)

                error_total += np.mean((y - salida)**2)

                # Conteo de aciertos según el modo de clasificación
                if modo == 'multiclase':
                    pos_pred = np.argmax(salida)
                    salida_clase = np.ones_like(salida) * -1
                    salida_clase[pos_pred] = 1
                    if np.array_equal(salida_clase, y):
                        aciertos += 1
                else:  # binario
                    prediccion = np.round(salida)
                    if np.array_equal(prediccion, y):
                        aciertos += 1

            error_total /= len(X)
            tasa_aciertos = aciertos / len(X)
            history["acc"].append(tasa_aciertos)
            history["ecm"].append(error_total)

            print(f"Época {epoca+1}: Aciertos={tasa_aciertos*100:.2f}%, ECM={error_total:.4f}")

            # Criterio de parada: error suficientemente pequeño
            if umbral_error is not None and error_total < umbral_error:
                print(f"Convergió en época {epoca+1} con error {error_total:.6f}")
                break

        return history

    def predecir(self, X):
        # Realiza predicciones para un conjunto de entradas.
        Y = []
        for x in X: # cada patrón de entrada
            salida = self.forward(x)
            Y.append(salida)
        return np.array(Y)
