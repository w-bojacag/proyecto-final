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

from . import almacen
from .config import settings

MODOS_DATOS = ("demo", "sqlite")
if settings.modo_datos not in MODOS_DATOS:
    raise RuntimeError(f"MODO_DATOS debe ser 'demo' o 'sqlite', no '{settings.modo_datos}'.")

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
    df["mes"] = df["semana"].dt.month
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

    df = df.sort_values(["municipio_id", "semana"]).reset_index(drop=True)
    # Etiqueta como en el pipeline: hay brote dentro de `horizonte_semanas` semanas (vacía al final de la serie).
    df["etiqueta"] = (
        df.groupby("municipio_id")["exceso"].shift(-settings.horizonte_semanas).astype(float)
    )
    return df


def serie_municipio(municipio_id: str) -> pd.DataFrame | None:
    """Serie del municipio según MODO_DATOS. None sólo en modo sqlite sin datos cargados."""
    if settings.modo_datos == "sqlite":
        return almacen.serie_semanal(municipio_id)
    df = _serie_completa()
    return df[df["municipio_id"] == municipio_id].reset_index(drop=True)


def rango_de_corte(primera_semana: pd.Timestamp, ultima_semana: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fechas de corte válidas: hay historial suficiente y existe la semana objetivo."""
    inicio = primera_semana + pd.Timedelta(weeks=settings.ventana_historial)
    fin = ultima_semana - pd.Timedelta(weeks=settings.horizonte_semanas)
    return inicio, fin


def rango_fechas_corte() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Fechas de corte válidas según MODO_DATOS. None si en sqlite no hay ningún municipio con datos.

    En sqlite se usa la parte común de los municipios cargados, para que una fecha válida lo sea para todos.
    """
    if settings.modo_datos == "sqlite":
        rangos = [r for r in (almacen.rango_datos(m) for m in MUNICIPIOS) if r is not None]
        if not rangos:
            return None
        return rango_de_corte(max(r[0] for r in rangos), min(r[1] for r in rangos))
    df = _serie_completa()
    return rango_de_corte(df["semana"].min(), df["semana"].max())
