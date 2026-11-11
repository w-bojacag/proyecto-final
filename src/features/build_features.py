"""
Script de Ingeniería de Características (Feature Engineering) para Alertas Tempranas de Dengue.
Objetivo: Generar variables de rezago (lags) y promedios móviles (rolling means) a partir 
de la matriz semanal integrada (ambiental + epidemiológica) de Cartagena.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_cruce_semanal_cartagena.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_caracteristicas_cartagena.csv"


def construir_caracteristicas():
    print("=" * 70)
    print("INGENIERÍA DE CARACTERÍSTICAS: REZAGOS (LAGS) Y VENTANAS MÓVILES")
    print("=" * 70)
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo base en: {INPUT_PATH}. Ejecute primero cruce_ambiental_epidemiologico.py")
    
    df = pd.read_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} semanas.")
    
    # Asegurar orden cronológico estricto
    df = df.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)

    # Imputar vacíos ambientales (p. ej. MODIS LST sin dato por nubosidad) con interpolación
    # lineal entre semanas vecinas, en lugar de eliminar semanas: un dropna() global borraba
    # ~270 semanas dispersas y rompía la continuidad de la serie.
    cols_ambientales = [
        "precipitation_mm", "lst_day_c", "lst_night_c", "air_temp_max_c",
        "air_temp_mean_c", "air_temp_min_c", "dewpoint_c", "evi", "ndvi"
    ]
    cols_ambientales = [col for col in cols_ambientales if col in df.columns]
    print("Vacíos ambientales antes de imputar:")
    print(df[cols_ambientales].isna().sum()[lambda s: s > 0].to_string())
    df[cols_ambientales] = df[cols_ambientales].interpolate(method="linear", limit_direction="both")

    # 1. Variables climáticas: rezagos amplios (ciclo biológico del vector: 4 a 12 semanas)
    cols_clima = [
        "precipitation_mm",
        "air_temp_mean_c",
        "lst_day_c",
        "dewpoint_c"
    ]
    cols_clima = [col for col in cols_clima if col in df.columns]
    
    lags_clima = [4, 6, 8, 12]
    for col in cols_clima:
        for lag in lags_clima:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    # Promedios móviles amplios para el clima (8 semanas, rezagados 4 semanas)
    for col in cols_clima:
        df[f"{col}_roll_mean_8"] = df[col].shift(4).rolling(window=8).mean()

    # 2. Variables epidemiológicas (casos): rezagos cortos y memoria anual (52 semanas)
    cols_casos = ["casos_confirmados_o_probables"]
    cols_casos = [col for col in cols_casos if col in df.columns]
    
    lags_casos = [4, 52]
    for col in cols_casos:
        for lag in lags_casos:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    # Promedio móvil de casos recientes
    for col in cols_casos:
        df[f"{col}_roll_mean_4"] = df[col].shift(1).rolling(window=4).mean()

    # 3. Corte estricto basado en el rezago máximo (52 semanas para la memoria anual)
    max_lag = max(max(lags_clima), max(lags_casos))
    df_modelado = df.iloc[max_lag:].reset_index(drop=True)
    
    if df_modelado.isna().any().any():
        raise ValueError("Quedan valores vacíos en la matriz de características")
    
    print(f"Dimensiones antes de eliminar NaNs por rezagos: {df.shape}")
    print(f"Dimensiones de la matriz final de características: {df_modelado.shape}")
    
    # Exportar resultado
    df_modelado.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMatriz de características guardada exitosamente en: {OUTPUT_PATH}")
    return df_modelado


if __name__ == "__main__":
    construir_caracteristicas()
