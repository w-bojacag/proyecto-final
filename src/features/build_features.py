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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rutas import cargar_params, municipio_desde_argumentos  # noqa: E402


def umbral_movil(df, col, params):
    """Canal endémico móvil: umbral de brote de la semana s del año Y calculado con los casos de
    la semana s en los años Y-anos_ventana a Y-1. Usa solo años anteriores (sin información futura).

    Métodos (params["metodo"]):
    - "percentil": percentil `percentil` de esos años (por defecto P75).
    - "bortman": límite superior del canal de Bortman (1999): intervalo de confianza del 95 %
      (t de Student) de la media geométrica. Los ceros se manejan sumando un desplazamiento
      mínimo (0,001), que según Umaña et al. (2026) rinde mejor que sumar 1.

    Devuelve el umbral (límite superior) y la línea central del canal: la media geométrica en
    Bortman y la mediana en el método de percentil. Entre ambas está la zona de alerta.

    - Si hay menos de `anos_minimos` años previos, el umbral queda vacío y esas semanas no se
      usan (antes se completaban con el percentil de toda la serie, que incluye años futuros).
    - La semana 53 existe solo en algunos años: se compara con la historia de la semana 52.
    """
    ventana, minimo = params["anos_ventana"], params["anos_minimos"]
    semana = df["SEMANA"].clip(upper=52)
    tabla = df[df["SEMANA"] <= 52].pivot(index="ANO", columns="SEMANA", values=col)
    tabla.columns.name = "SEMANA_REF"
    tabla = tabla.reindex(range(tabla.index.min(), tabla.index.max() + 1))
    if params["metodo"] == "percentil":
        umbrales = tabla.rolling(ventana, min_periods=minimo).quantile(params["percentil"]).shift(1)
        centros = tabla.rolling(ventana, min_periods=minimo).quantile(0.5).shift(1)
    elif params["metodo"] == "bortman":
        desplazamiento = params["bortman_desplazamiento"]
        logs = np.log(tabla + desplazamiento)
        media = logs.rolling(ventana, min_periods=minimo).mean()
        desv = logs.rolling(ventana, min_periods=minimo).std()
        n = logs.rolling(ventana, min_periods=minimo).count()
        t = pd.DataFrame(stats.t.ppf(0.975, n - 1), index=n.index, columns=n.columns)
        umbrales = (np.exp(media + t * desv / np.sqrt(n)) - desplazamiento).clip(lower=0).shift(1)
        centros = (np.exp(media) - desplazamiento).clip(lower=0).shift(1)
    else:
        raise ValueError(f"Método de canal endémico desconocido: {params['metodo']}")
    largo = pd.DataFrame({"umbral": umbrales.stack(), "centro": centros.stack()}).reset_index()
    largo.columns = ["ANO", "SEMANA_REF", "umbral", "centro"]
    unido = df[["ANO"]].assign(SEMANA_REF=semana.values).merge(largo, on=["ANO", "SEMANA_REF"], how="left")
    return unido["umbral"].values, unido["centro"].values


def agregar_etiqueta(df, params, col="casos_confirmados_o_probables"):
    """Umbral de brote, etiqueta a T+h y variables relativas al umbral.

    Brote en la semana t: los casos superan el umbral durante `semanas_consecutivas` semanas
    seguidas que terminan en t. Con 1 basta una semana; con 3 se exige un exceso sostenido,
    como la definición de brote del Ministerio de Salud e INS usada por Umaña et al. (2026),
    lo que evita que un caso aislado en una semana de umbral bajo cuente como brote.
    """
    h = params["horizonte"]
    df = df.copy()
    df["umbral_brote"], df["centro_canal"] = umbral_movil(df, col, params)
    supera = (df[col] > df["umbral_brote"]).astype(int)
    k = params["semanas_consecutivas"]
    df["es_brote_actual"] = (supera.rolling(k, min_periods=k).min() == 1).astype(int)
    df["target_brote_t4"] = df["es_brote_actual"].shift(-h)

    # El umbral de la semana objetivo (t+h) solo depende de años anteriores, así que se conoce en t.
    # Los modelos comparan los casos actuales contra ese umbral: el volumen absoluto por sí solo
    # no sirve, porque el umbral sube después de cada año epidémico.
    df["umbral_objetivo"] = df["umbral_brote"].shift(-h)
    df["ratio_casos_umbral_objetivo"] = df[col] / (df["umbral_objetivo"] + 1.0)
    df["ratio_roll4_umbral_objetivo"] = df[f"{col}_roll_mean_4"] / (df["umbral_objetivo"] + 1.0)
    df["exceso_actual"] = df[col] / (df["umbral_brote"] + 1.0)
    return df


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
    h = params["horizonte"]
    df = agregar_etiqueta(df, params)

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
