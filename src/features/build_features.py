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
    
    # Variables a las que les aplicaremos rezagos (lags) y medias móviles
    # - Clima: precipitación, temperatura media, humedad/punto de rocío
    # - Epidemiología: casos confirmados o probables
    cols_a_rezagar = [
        "precipitation_mm",
        "air_temp_mean_c",
        "lst_day_c",
        "dewpoint_c",
        "casos_confirmados_o_probables"
    ]
    
    lags = [1, 2, 3, 4]  # Rezagos de 1 a 4 semanas
    
    for col in cols_a_rezagar:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                
    # Promedios móviles (Rolling means) de 4 semanas para suavizar estacionalidad climática
    rolling_cols = ["precipitation_mm", "air_temp_mean_c", "casos_confirmados_o_probables"]
    for col in rolling_cols:
        if col in df.columns:
            df[f"{col}_roll_mean_4"] = df[col].shift(1).rolling(window=4).mean()
            
    # Eliminar filas con valores NaN generados por los rezagos iniciales (primeras semanas)
    df_modelado = df.dropna().reset_index(drop=True)
    
    print(f"Dimensiones antes de eliminar NaNs por rezagos: {df.shape}")
    print(f"Dimensiones de la matriz final de características: {df_modelado.shape}")
    
    # Exportar resultado
    df_modelado.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMatriz de características guardada exitosamente en: {OUTPUT_PATH}")
    return df_modelado


if __name__ == "__main__":
    construir_caracteristicas()
