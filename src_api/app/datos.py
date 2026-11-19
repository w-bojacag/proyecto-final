"""Serie semanal por municipio.

SIMULADA: genera datos sintéticos con la misma estructura que tendrá el dato real
(casos SIVIGILA semanales, lluvia y temperatura desde Google Earth Engine).
No son datos reales ni resultados de ningún modelo.

Para usar datos reales: reemplazar el cuerpo de `_serie_completa()` por la lectura
del archivo procesado (columnas: municipio_id, semana, casos, lluvia_mm,
temp_media_c, umbral_canal, exceso). El resto de la API no cambia.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import settings

MUNICIPIOS: dict[str, str] = {
    "cartagena": "Cartagena",
    "magangue": "Magangué",
    "el_carmen_de_bolivar": "El Carmen de Bolívar",
    "santa_rosa_del_sur": "Santa Rosa del Sur",
    "turbaco": "Turbaco",
}


@lru_cache(maxsize=1)
def _serie_completa() -> pd.DataFrame:
    rng = np.random.default_rng(settings.semilla)
    semanas = pd.date_range(
        f"{settings.anio_inicio}-01-07", f"{settings.anio_fin}-12-31", freq="W-SUN"
    )
    dia = semanas.dayofyear.to_numpy()
    estacional = np.sin(2 * np.pi * dia / 365.25)

    frames = []
    for i, mid in enumerate(MUNICIPIOS):
        lluvia = np.clip(
            60 + 45 * np.sin(2 * np.pi * dia / 365.25 - 0.8 + 0.2 * i)
            + rng.normal(0, 12, len(semanas)),
            0,
            None,
        )
        temp = 27.5 + 1.2 * np.cos(2 * np.pi * dia / 365.25) + rng.normal(0, 0.4, len(semanas))
        lluvia_lag4 = pd.Series(lluvia).shift(4).bfill().to_numpy()
        lam = (12 + 6 * i) * np.exp(0.5 * estacional + 0.006 * (lluvia_lag4 - 60))
        frames.append(
            pd.DataFrame(
                {
                    "municipio_id": mid,
                    "semana": semanas,
                    "casos": rng.poisson(lam),
                    "lluvia_mm": lluvia.round(1),
                    "temp_media_c": temp.round(1),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["anio"] = df["semana"].dt.year
    df["semana_anio"] = df["semana"].dt.isocalendar().week.clip(upper=52).astype(int)

    # Canal endémico simulado: percentil 75 por semana del año, usando solo años de
    # entrenamiento. Método pendiente de justificar (ver SesionesConExperto).
    entrenamiento = df[df["anio"] <= settings.anio_entrenamiento_fin]
    p75 = (
        entrenamiento.groupby(["municipio_id", "semana_anio"])["casos"]
        .quantile(0.75)
        .rename("umbral_canal")
        .reset_index()
    )
    df = df.merge(p75, on=["municipio_id", "semana_anio"], how="left")
    df["umbral_canal"] = df["umbral_canal"].fillna(entrenamiento["casos"].quantile(0.75))
    df["exceso"] = df["casos"] > df["umbral_canal"]

    return df.sort_values(["municipio_id", "semana"]).reset_index(drop=True)


def serie_municipio(municipio_id: str) -> pd.DataFrame:
    df = _serie_completa()
    return df[df["municipio_id"] == municipio_id].reset_index(drop=True)


def rango_fechas_corte() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fechas de corte válidas: hay historial suficiente y existe la semana objetivo."""
    df = _serie_completa()
    inicio = df["semana"].min() + pd.Timedelta(weeks=settings.ventana_historial)
    fin = df["semana"].max() - pd.Timedelta(weeks=settings.horizonte_semanas)
    return inicio, fin
