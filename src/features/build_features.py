"""
Script de Ingeniería de Características (Feature Engineering) para Alertas Tempranas de Dengue.
Objetivo: Generar variables de rezago (lags) y promedios móviles (rolling means) a partir 
de la matriz semanal integrada (ambiental + epidemiológica) de un municipio.
Uso: python src/features/build_features.py --municipio cartagena
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rutas import cargar_params, municipio_desde_argumentos  # noqa: E402


def umbral_movil(df, col, anos_ventana, percentil, anos_minimos):
    """Canal endémico móvil: para la semana s del año Y, percentil de los casos de la semana s
    en los años Y-anos_ventana a Y-1. Usa solo años anteriores (sin información futura).

    - Si hay menos de `anos_minimos` años previos, el umbral queda vacío y esas semanas no se
      usan (antes se completaban con el percentil de toda la serie, que incluye años futuros).
    - La semana 53 existe solo en algunos años: se compara con la historia de la semana 52.
    """
    semana = df["SEMANA"].clip(upper=52)
    tabla = df[df["SEMANA"] <= 52].pivot(index="ANO", columns="SEMANA", values=col)
    tabla.columns.name = "SEMANA_REF"
    tabla = tabla.reindex(range(tabla.index.min(), tabla.index.max() + 1))
    umbrales = tabla.rolling(anos_ventana, min_periods=anos_minimos).quantile(percentil).shift(1)
    largo = umbrales.stack().rename("umbral").reset_index()
    largo.columns = ["ANO", "SEMANA_REF", "umbral"]
    return (
        df[["ANO"]].assign(SEMANA_REF=semana.values)
        .merge(largo, on=["ANO", "SEMANA_REF"], how="left")["umbral"].values
    )


def construir_caracteristicas(rutas):
    print("=" * 70)
    print("INGENIERÍA DE CARACTERÍSTICAS: REZAGOS (LAGS) Y VENTANAS MÓVILES")
    print("=" * 70)
    
    if not rutas.cruce.exists():
        raise FileNotFoundError(f"No se encontró el archivo base en: {rutas.cruce}. Ejecute primero cruce_ambiental_epidemiologico.py")
    
    df = pd.read_csv(rutas.cruce)
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

    # 2. Variables epidemiológicas (casos): Tasa de cambio (velocidad) y memoria anual
    cols_casos = ["casos_confirmados_o_probables"]
    cols_casos = [col for col in cols_casos if col in df.columns]

    for col in cols_casos:
        # A. Tasa de cambio a 4 semanas (velocidad de brote a mediano plazo)
        df[f"{col}_tasa_cambio_4"] = (df[col] - df[col].shift(4)) / (df[col].shift(4) + 1.0)
        
        # B. Tasa de cambio a corto plazo (velocidad inmediata de 1 semana)
        df[f"{col}_tasa_cambio_1"] = (df[col] - df[col].shift(1)) / (df[col].shift(1) + 1.0)
        
        # C. Memoria anual normalizada (estacionalidad de año a año)
        df[f"{col}_lag_52"] = df[col].shift(52)
            
    # Promedio móvil de casos recientes
    for col in cols_casos:
        df[f"{col}_roll_mean_4"] = df[col].shift(1).rolling(window=4).mean()

    # 3. Componentes de Estacionalidad (Tiempo)
    # Convertimos la semana epidemiológica a mes y trimestre aproximado
    df["mes_aproximado"] = np.ceil(df["SEMANA"] / 4.33).astype(int)
    df["trimestre_aproximado"] = np.ceil(df["SEMANA"] / 13.0).astype(int)

    # 4. Canal endémico móvil, etiqueta de brote a T+h y variables relativas al umbral
    params = cargar_params("target")
    col = "casos_confirmados_o_probables"
    h = params["horizonte"]
    df["umbral_brote"] = umbral_movil(
        df, col, params["anos_ventana"], params["percentil"], params["anos_minimos"]
    )
    df["es_brote_actual"] = (df[col] > df["umbral_brote"]).astype(int)
    df["target_brote_t4"] = df["es_brote_actual"].shift(-h)

    # El umbral de la semana objetivo (t+h) solo depende de años anteriores, así que se conoce en t.
    # Los modelos comparan los casos actuales contra ese umbral: el volumen absoluto por sí solo
    # no sirve, porque el umbral sube después de cada año epidémico.
    df["umbral_objetivo"] = df["umbral_brote"].shift(-h)
    df["ratio_casos_umbral_objetivo"] = df[col] / (df["umbral_objetivo"] + 1.0)
    df["ratio_roll4_umbral_objetivo"] = df[f"{col}_roll_mean_4"] / (df["umbral_objetivo"] + 1.0)
    df["exceso_actual"] = df[col] / (df["umbral_brote"] + 1.0)

    # 5. Corte estricto basado en el rezago máximo (52 semanas para la memoria anual) y en las
    # últimas h semanas, cuya etiqueta futura todavía no se conoce.
    # También se descartan las primeras semanas sin historia suficiente para el umbral.
    max_lag = max(max(lags_clima), 52)  # 52 semanas es nuestro rezago máximo (memoria anual)
    df_modelado = df.iloc[max_lag:len(df) - h]
    df_modelado = df_modelado.dropna(subset=["umbral_brote", "umbral_objetivo"]).reset_index(drop=True)
    print(f"Primera semana con umbral: {df_modelado['ANO'].iloc[0]}-S{df_modelado['SEMANA'].iloc[0]}")

    if df_modelado.isna().any().any():
        vacias = df_modelado.columns[df_modelado.isna().any()].tolist()
        raise ValueError(f"Quedan valores vacíos en la matriz de características: {vacias}")
    print("Proporción de semanas con brote (T+h) por año:")
    print(df_modelado.groupby("ANO")["target_brote_t4"].mean().round(2).to_string())
    
    print(f"Dimensiones antes de eliminar NaNs por rezagos: {df.shape}")
    print(f"Dimensiones de la matriz final de características: {df_modelado.shape}")
    
    # Exportar resultado
    df_modelado.to_csv(rutas.caracteristicas, index=False)
    print(f"\nMatriz de características guardada exitosamente en: {rutas.caracteristicas}")
    return df_modelado


if __name__ == "__main__":
    construir_caracteristicas(municipio_desde_argumentos("Matriz de características de un municipio"))
