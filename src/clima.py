"""
Extracción del clima diario 2007-2025 desde Google Earth Engine para la cabecera urbana de un
municipio (polígono DANE MGN 2021 en data/external).

Uso: python src/clima.py --municipio cartagena
"""

import json

import ee
import pandas as pd

from rutas import municipio_desde_argumentos

ANOS = range(2007, 2026)

# CHIRPS (~5,5 km) y ERA5-Land (~11 km) tienen píxeles grandes frente a la ciudad: se muestrean
# a 1 km para que el promedio pondere cada píxel por el área que comparte con el polígono
# (a escala nativa solo cuentan los píxeles cuyo centro cae dentro y el valor puede quedar vacío).
ESCALA_GRUESA = 1000


def _media(img, banda, poligono, escala):
    return img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono, scale=escala).get(banda)


def _coleccion(nombre, bandas, year):
    return ee.ImageCollection(nombre).filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(bandas)


def _a_dataframe(col, map_func):
    return pd.DataFrame([f["properties"] for f in col.map(map_func).getInfo()["features"]])


def extraer_chirps(year, poligono):
    """1. Precipitación diaria (CHIRPS)"""
    def map_func(img):
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"),
                                 "precipitation_mm": _media(img, "precipitation", poligono, ESCALA_GRUESA)})
    return _a_dataframe(_coleccion("UCSB-CHG/CHIRPS/DAILY", "precipitation", year), map_func)


def extraer_modis_lst(year, poligono):
    """2. Temperatura superficial día y noche (MODIS LST, diaria)"""
    def map_func(img):
        day = _media(img, "LST_Day_1km", poligono, 1000)
        night = _media(img, "LST_Night_1km", poligono, 1000)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "lst_day_c": ee.Algorithms.If(day, ee.Number(day).multiply(0.02).subtract(273.15), None),
            "lst_night_c": ee.Algorithms.If(night, ee.Number(night).multiply(0.02).subtract(273.15), None),
        })
    return _a_dataframe(_coleccion("MODIS/061/MOD11A1", ["LST_Day_1km", "LST_Night_1km"], year), map_func)


def extraer_era5_clima(year, poligono):
    """3. Temperatura del aire (media, mínima, máxima) y punto de rocío (ERA5-Land, diaria)"""
    bandas = {
        "air_temp_mean_c": "temperature_2m", "air_temp_min_c": "temperature_2m_min",
        "air_temp_max_c": "temperature_2m_max", "dewpoint_c": "dewpoint_temperature_2m",
    }

    def map_func(img):
        props = {"date": img.date().format("YYYY-MM-dd")}
        for salida, banda in bandas.items():
            v = _media(img, banda, poligono, ESCALA_GRUESA)
            props[salida] = ee.Algorithms.If(v, ee.Number(v).subtract(273.15), None)
        return ee.Feature(None, props)
    return _a_dataframe(_coleccion("ECMWF/ERA5_LAND/DAILY_AGGR", list(bandas.values()), year), map_func)


def extraer_modis_vegetacion(year, poligono):
    """4. Índices de vegetación NDVI y EVI (MODIS, 16 días)"""
    def map_func(img):
        ndvi = _media(img, "NDVI", poligono, 250)
        evi = _media(img, "EVI", poligono, 250)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "ndvi": ee.Algorithms.If(ndvi, ee.Number(ndvi).multiply(0.0001), None),
            "evi": ee.Algorithms.If(evi, ee.Number(evi).multiply(0.0001), None),
        })
    return _a_dataframe(_coleccion("MODIS/061/MOD13Q1", ["NDVI", "EVI"], year), map_func)


def main():
    rutas = municipio_desde_argumentos("Extrae el clima diario de un municipio desde Earth Engine")
    ee.Initialize(project="dengue-project-2026")
    with open(rutas.poligono, encoding="utf-8") as f:
        poligono = ee.Geometry(json.load(f)["geometry"])

    print(f"Extrayendo clima de {rutas.nombre} ({rutas.poligono.name})...")
    fuentes = {"chirps": extraer_chirps, "lst": extraer_modis_lst,
               "era5": extraer_era5_clima, "veg": extraer_modis_vegetacion}
    partes = {k: [] for k in fuentes}
    for anio in ANOS:
        print(f"  {anio}")
        for k, funcion in fuentes.items():
            partes[k].append(funcion(anio, poligono))

    tablas = {}
    for k, lista in partes.items():
        df = pd.concat(lista, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        tablas[k] = df.set_index("date")

    # Base diaria completa; NDVI/EVI (cada 16 días) se propaga hacia adelante entre imágenes
    maestro = pd.DataFrame(index=pd.date_range("2007-01-01", "2025-12-31", freq="D"))
    maestro.index.name = "date"
    maestro = maestro.join(tablas["chirps"]).join(tablas["lst"]).join(tablas["era5"])
    veg = tablas["veg"][~tablas["veg"].index.duplicated(keep="first")]
    maestro = maestro.join(veg.reindex(maestro.index).ffill())

    maestro.to_csv(rutas.clima)
    print(f"Clima guardado en: {rutas.clima} ({len(maestro)} días)")
    print(maestro.isna().sum().to_string())


if __name__ == "__main__":
    main()
