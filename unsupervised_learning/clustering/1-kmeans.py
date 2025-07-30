import numpy as np


def initialize(X, k):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    return np.random.uniform(X.min(axis=0), X.max(axis=0), size=(k, d))


def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        C_prev = C.copy()

        # Bucle 1: asignar cada punto al centroide más cercano
        for i in range(n):
            # distancia euclidiana al cuadrado
            distances = np.sum((C - X[i])**2, axis=1)
            clss[i] = np.argmin(distances)

        # Bucle 2: actualizar centroides
        for cluster_idx in range(k):
            points = X[clss == cluster_idx]
            if len(points) == 0:
                # Re-inicializar centroide vacío
                C[cluster_idx] =\
                    np.random.uniform(X.min(axis=0), X.max(axis=0))
            else:
                C[cluster_idx] = points.mean(axis=0)

        # Verificar convergencia
        if np.allclose(C, C_prev):
            break

    return C, clss
