"""
Script independiente para el cruce de información ambiental y epidemiológica (Cartagena, 2007-2025).
Objetivo:
1. Cargar la matriz maestra ambiental diaria de Cartagena.
2. Cargar los datos epidemiológicos de SIVIGILA (Bolívar) y filtrar para Cartagena.
3. Agregar los datos ambientales diarios a nivel semanal (coincidiendo con las semanas epidemiológicas).
4. Agregar los casos de dengue de SIVIGILA a nivel semanal (por ANO y SEMANA).
5. Unir (merge) ambas series en una única matriz integrada para modelado de alerta temprana.
6. Exportar el resultado a CSV.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Definición de rutas relativas
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
AMBIENTAL_PATH = ROOT_DIR / "data" / "processed" / "matriz_maestra_ambiental_cartagena_2007_2025.csv"
SIVIGILA_PATH = ROOT_DIR / "data" / "processed" / "dengue_bolivar_unificado.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_cruce_semanal_cartagena.csv"


def _parse_fecha(serie):
    """Parsea fechas de SIVIGILA con formato 'DD/MM/YYYY HH:MM:SS'."""
    solo_fecha = serie.astype(str).str.split(" ").str[0]
    return pd.to_datetime(solo_fecha, format="%d/%m/%Y", errors="coerce")


def cargar_y_procesar_ambiental():
    print("=" * 70)
    print("1. CARGANDO Y PROCESANDO MATRIZ AMBIENTAL DIARIA")
    print("=" * 70)
    df_env = pd.read_csv(AMBIENTAL_PATH)
    df_env["date"] = pd.to_datetime(df_env["date"], errors="coerce")
    
    # Extraer año y semana ISO para alinear con SIVIGILA
    iso_cal = df_env["date"].dt.isocalendar()
    df_env["ANO"] = iso_cal.year
    df_env["SEMANA"] = iso_cal.week
    
    # Manejo especial para semanas ISO 53 en transición de año o año bisiesto si fuera necesario
    # Agregación semanal:
    # - precipitation_mm: suma total acumulada en la semana
    # - variables de temperatura y vegetación: promedio semanal
    agregaciones = {
        "precipitation_mm": "sum",
        "lst_day_c": "mean",
        "lst_night_c": "mean",
        "air_temp_max_c": "mean",
        "air_temp_mean_c": "mean",
        "air_temp_min_c": "mean",
        "dewpoint_c": "mean",
        "evi": "mean",
        "ndvi": "mean"
    }
    
    df_env_semanal = df_env.groupby(["ANO", "SEMANA"]).agg(agregaciones).reset_index()
    print(f"Registros ambientales diarios: {len(df_env)}")
    print(f"Registros ambientales semanales agregados: {len(df_env_semanal)}")
    return df_env_semanal


def cargar_y_procesar_epidemiologico():
    print("\n" + "=" * 70)
    print("2. CARGANDO Y PROCESANDO DATOS SIVIGILA (CARTAGENA)")
    print("=" * 70)
    df_epi = pd.read_csv(SIVIGILA_PATH, low_memory=False)
    
    # Filtrar estrictamente para Cartagena de Indias
    # Verificamos los nombres en 'Municipio_ocurrencia'
    if "Municipio_ocurrencia" in df_epi.columns:
        # Cartagena suele venir como 'CARTAGENA'
        df_cartagena = df_epi[df_epi["Municipio_ocurrencia"].str.upper().str.contains("CARTAGENA", na=False)].copy()
    else:
        raise ValueError("No se encontró la columna 'Municipio_ocurrencia' en el dataset de SIVIGILA.")
    
    print(f"Casos totales en Bolívar: {len(df_epi):,}")
    print(f"Casos filtrados para Cartagena: {len(df_cartagena):,}")
    
    # Agregación semanal de casos por ANO y SEMANA
    # Podemos segregar por tipo de caso (ej. todos, o solo confirmados TIP_CAS=3, o confirmados+probables TIP_CAS in [2,3])
    # Aquí creamos conteos generales y específicos para análisis posterior.
    df_cartagena["casos_totales"] = 1
    df_cartagena["casos_confirmados"] = (df_cartagena["TIP_CAS"] == 3).astype(int)
    df_cartagena["casos_probables"] = (df_cartagena["TIP_CAS"] == 2).astype(int)
    df_cartagena["casos_confirmados_o_probables"] = df_cartagena["TIP_CAS"].isin([2, 3]).astype(int)
    
    df_epi_semanal = (
        df_cartagena.groupby(["ANO", "SEMANA"])
        .agg({
            "casos_totales": "sum",
            "casos_confirmados": "sum",
            "casos_probables": "sum",
            "casos_confirmados_o_probables": "sum"
        })
        .reset_index()
    )
    
    print(f"Semanas epidemiológicas con registros en Cartagena: {len(df_epi_semanal)}")
    return df_epi_semanal


def fusionar_datos(df_env_semanal, df_epi_semanal):
    print("\n" + "=" * 70)
    print("3. FUSIONANDO MATRIZ AMBIENTAL Y EPIDEMIOLÓGICA")
    print("=" * 70)
    
    # Merge por ANO y SEMANA (Outer join para conservar la línea de tiempo completa ambiental y epidemiológica)
    df_master = pd.merge(df_env_semanal, df_epi_semanal, on=["ANO", "SEMANA"], how="outer")
    
    # Ordenar cronológicamente
    df_master = df_master.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
    
    # Rellenar con 0 los casos en semanas donde no hubo reportes de dengue
    casos_cols = ["casos_totales", "casos_confirmados", "casos_probables", "casos_confirmados_o_probables"]
    for col in casos_cols:
        if col in df_master.columns:
            df_master[col] = df_master[col].fillna(0)
            
    print(f"Dimensiones de la matriz maestra integrada: {df_master.shape}")
    print("\nPrimeras filas del resultado:")
    print(df_master.head(10).to_string())
    
    # Guardar a CSV
    df_master.to_csv(OUT_PATH, index=False)
    print(f"\nMatriz integrada guardada exitosamente en: {OUT_PATH}")
    return df_master


def main():
    df_env_sem = cargar_y_procesar_ambiental()
    df_epi_sem = cargar_y_procesar_epidemiologico()
    df_master = fusionar_datos(df_env_sem, df_epi_sem)
    print("\n¡Proceso de cruce completado con éxito!")


if __name__ == "__main__":
    main()
