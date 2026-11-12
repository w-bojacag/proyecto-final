"""
Generador de ventanas temporales (tensores 3D) para el modelo LSTM.

Cada muestra corresponde a una semana t de la matriz de características: la entrada son las
window_size semanas que terminan en t y la etiqueta es la de esa fila (brote en t+4, ya
desplazada en build_features.py). Así el LSTM y XGBoost predicen exactamente las mismas semanas.
"""

import numpy as np


def crear_ventanas_temporales(valores, window_size):
    """
    Args:
        valores: array 2D [semanas, variables] en orden cronológico y sin vacíos.
        window_size: semanas de historia que ve la red.

    Returns:
        X: array 3D [muestras, window_size, variables]; la muestra j termina en la fila
           j + window_size - 1 de `valores`.
    """
    n = len(valores) - window_size + 1
    return np.stack([valores[i:i + window_size] for i in range(n)])
