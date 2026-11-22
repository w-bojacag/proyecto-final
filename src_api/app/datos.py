"""Serie semanal por municipio, leída de la base SQLite.

Los datos se cargan con `POST /api/v1/datos/{municipio_id}/csv` desde `caracteristicas.csv`
del pipeline (ver almacen.py). Sin datos cargados, los endpoints de serie y predicción responden 404.
"""
import pandas as pd

from . import almacen
from .config import settings

MUNICIPIOS: dict[str, str] = {
    "cartagena": "Cartagena",
    "magangue": "Magangué",
    "el_carmen_de_bolivar": "El Carmen de Bolívar",
    "santa_rosa_del_sur": "Santa Rosa del Sur",
    "turbaco": "Turbaco",
}


def serie_municipio(municipio_id: str) -> pd.DataFrame | None:
    """Serie del municipio desde SQLite. None si el municipio no tiene datos cargados."""
    return almacen.serie_semanal(municipio_id)


def rango_de_corte(
    primera_semana: pd.Timestamp, ultima_semana: pd.Timestamp, ventana_historial: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fechas de corte válidas: hay `ventana_historial` semanas de historial y existe la semana objetivo."""
    inicio = primera_semana + pd.Timedelta(weeks=ventana_historial)
    fin = ultima_semana - pd.Timedelta(weeks=settings.horizonte_semanas)
    return inicio, fin


def rango_fechas_corte(ventana_historial: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Fechas de corte válidas. None si no hay ningún municipio con datos.

    Se usa la parte común de los municipios cargados, para que una fecha válida lo sea para todos.
    """
    rangos = [r for r in (almacen.rango_datos(m) for m in MUNICIPIOS) if r is not None]
    if not rangos:
        return None
    return rango_de_corte(max(r[0] for r in rangos), min(r[1] for r in rangos), ventana_historial)
