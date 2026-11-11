"""
Generador de ventanas temporales (Tensores 3D) para el modelo LSTM.
Convierte los datos tabulares secuenciales de la matriz de cruce en 
bloques de tiempo [samples, timesteps, features].
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
# Usamos la matriz de cruce base (antes de los lags artificiales del XGBoost)
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_cruce_semanal_cartagena.csv"

def crear_ventanas_temporales(df, columnas_features, columna_target, window_size, horizonte=4):
    """
    Crea ventanas temporales deslizantes.
    
    Args:
        df: DataFrame original ordenado cronológicamente.
        columnas_features: Lista de columnas predictoras.
        columna_target: Nombre de la columna objetivo.
        window_size: Tamaño de la ventana (ej. 8 semanas mirando hacia atrás).
        horizonte: Semanas hacia adelante para predecir (T+4).
        
    Returns:
        X (numpy array 3D): [muestras, window_size, n_features]
        y (numpy array 1D): [muestras]
        fechas (pd.Series o list): Referencia temporal de la predicción.
    """
    X, y, indices = [], [], []
    
    # Asegurar que no hay NaNs en las features. Imputar si es necesario.
    df_features = df[columnas_features].interpolate(method='linear', limit_direction='both').fillna(0)
    features_array = df_features.values
    target_array = df[columna_target].values
    
    for i in range(len(df) - window_size - horizonte + 1):
        # La ventana de observación: desde i hasta i+window_size-1
        ventana_X = features_array[i : i + window_size, :].copy()
        
        # Normalización local por ventana (Z-score por cada variable para combatir Data Shift)
        # Se suma un epsilon (1e-8) para evitar división por cero en variables sin variación temporal
        mean = np.mean(ventana_X, axis=0)
        std = np.std(ventana_X, axis=0) + 1e-8
        ventana_X = (ventana_X - mean) / std
        
        # El objetivo (T+4): la posición de la última semana de la ventana + horizonte
        target_y = target_array[i + window_size + horizonte - 1]
        
        X.append(ventana_X)
        y.append(target_y)
        # Guardamos el índice (o año/semana) del momento de la predicción
        indices.append(i + window_size + horizonte - 1)
        
    return np.array(X), np.array(y), df.iloc[indices][["ANO", "SEMANA"]].reset_index(drop=True)
