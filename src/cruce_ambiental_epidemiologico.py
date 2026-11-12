"""
Cruce de información ambiental y epidemiológica de un municipio (2007-2025).
Uso: python src/cruce_ambiental_epidemiologico.py --municipio cartagena

Objetivo:
1. Cargar el clima diario del municipio.
2. Cargar los datos epidemiológicos de SIVIGILA (Bolívar) y filtrar por municipio de ocurrencia.
3. Agregar los datos ambientales diarios a nivel semanal (coincidiendo con las semanas epidemiológicas).
4. Agregar los casos de dengue de SIVIGILA a nivel semanal (por ANO y SEMANA).
5. Unir (merge) ambas series en una única matriz integrada para modelado de alerta temprana.
6. Exportar el resultado a CSV.
"""

import pandas as pd

from rutas import SIVIGILA_PATH, municipio_desde_argumentos


def _parse_fecha(serie):
    """Parsea fechas de SIVIGILA con formato 'DD/MM/YYYY HH:MM:SS'."""
    solo_fecha = serie.astype(str).str.split(" ").str[0]
    return pd.to_datetime(solo_fecha, format="%d/%m/%Y", errors="coerce")


def semana_epidemiologica(fechas):
    """
    Año y semana epidemiológica (INS/SIVIGILA, criterio MMWR) de una serie de fechas.
    La semana va de domingo a sábado y la semana 1 es la primera con al menos 4 días
    en enero; equivale a asignar cada semana al año en que cae su miércoles.
    (No usar isocalendar(): la semana ISO va de lunes a domingo y en años como 2009
    y 2015 queda desplazada una semana completa respecto a la epidemiológica.)
    """
    dias_desde_domingo = (fechas.dt.dayofweek + 1) % 7
    miercoles = fechas - pd.to_timedelta(dias_desde_domingo, unit="D") + pd.Timedelta(days=3)
    anio = miercoles.dt.year
    semana = (miercoles.dt.dayofyear - 1) // 7 + 1
    return anio, semana


def cargar_y_procesar_ambiental(rutas):
    print("=" * 70)
    print("1. CARGANDO Y PROCESANDO MATRIZ AMBIENTAL DIARIA")
    print("=" * 70)
    df_env = pd.read_csv(rutas.clima)
    df_env["date"] = pd.to_datetime(df_env["date"], errors="coerce")

    # Año y semana epidemiológica (domingo a sábado), igual que la SEMANA de SIVIGILA
    df_env["ANO"], df_env["SEMANA"] = semana_epidemiologica(df_env["date"])

    # Agregación semanal:
    # - precipitation_mm: acumulado semanal = promedio diario x 7. En semanas completas es
    #   igual a la suma; en las semanas de borde (2007-S1 empieza el 31/12/2006 y 2025-S53
    #   termina el 03/01/2026, fuera de la descarga) evita subestimar la lluvia.
    # - variables de temperatura y vegetación: promedio semanal
    agregaciones = {
        "precipitation_mm": "mean",
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
    df_env_semanal["precipitation_mm"] = df_env_semanal["precipitation_mm"] * 7
    print(f"Registros ambientales diarios: {len(df_env)}")
    print(f"Registros ambientales semanales agregados: {len(df_env_semanal)}")
    return df_env_semanal


def cargar_y_procesar_epidemiologico(rutas):
    print("\n" + "=" * 70)
    print(f"2. CARGANDO Y PROCESANDO DATOS SIVIGILA ({rutas.nombre})")
    print("=" * 70)
    df_epi = pd.read_csv(SIVIGILA_PATH, low_memory=False)
    
    # Municipio de ocurrencia (nombre exacto de SIVIGILA, definido en params.yaml)
    if "Municipio_ocurrencia" not in df_epi.columns:
        raise ValueError("No se encontró la columna 'Municipio_ocurrencia' en el dataset de SIVIGILA.")
    df_mun = df_epi[df_epi["Municipio_ocurrencia"].str.strip().str.upper() == rutas.nombre].copy()
    if df_mun.empty:
        raise ValueError(f"No hay registros de SIVIGILA para '{rutas.nombre}'")
    
    print(f"Casos totales en Bolívar: {len(df_epi):,}")
    print(f"Casos filtrados para {rutas.nombre}: {len(df_mun):,}")
    
    # Agregación semanal de casos por ANO y SEMANA
    # Podemos segregar por tipo de caso (ej. todos, o solo confirmados TIP_CAS=3, o confirmados+probables TIP_CAS in [2,3])
    # Aquí creamos conteos generales y específicos para análisis posterior.
    df_mun["casos_totales"] = 1
    df_mun["casos_confirmados"] = (df_mun["TIP_CAS"] == 3).astype(int)
    df_mun["casos_probables"] = (df_mun["TIP_CAS"] == 2).astype(int)
    df_mun["casos_confirmados_o_probables"] = df_mun["TIP_CAS"].isin([2, 3]).astype(int)
    
    df_epi_semanal = (
        df_mun.groupby(["ANO", "SEMANA"])
        .agg({
            "casos_totales": "sum",
            "casos_confirmados": "sum",
            "casos_probables": "sum",
            "casos_confirmados_o_probables": "sum"
        })
        .reset_index()
    )
    
    print(f"Semanas epidemiológicas con registros en {rutas.nombre}: {len(df_epi_semanal)}")
    return df_epi_semanal


def fusionar_datos(df_env_semanal, df_epi_semanal, rutas):
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
    df_master.to_csv(rutas.cruce, index=False)
    print(f"\nMatriz integrada guardada exitosamente en: {rutas.cruce}")
    return df_master


def main():
    rutas = municipio_desde_argumentos("Cruce semanal de clima y casos de dengue de un municipio")
    df_env_sem = cargar_y_procesar_ambiental(rutas)
    df_epi_sem = cargar_y_procesar_epidemiologico(rutas)
    fusionar_datos(df_env_sem, df_epi_sem, rutas)
    print("\n¡Proceso de cruce completado con éxito!")


if __name__ == "__main__":
    main()
