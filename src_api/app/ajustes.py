"""Parámetros ajustables desde el dashboard, guardados en SQLite (tabla `configuracion`).

Prioridad: el valor guardado en SQLite gana sobre la variable de entorno. Si nunca se guardó un
parámetro, se usa la variable de entorno (o su valor por defecto de config.py).
"""
from . import almacen
from .config import settings


def valores_por_defecto() -> dict:
    return {
        "umbral_decision": settings.umbral_decision,
        "anio_minimo_graficos": settings.anio_minimo_graficos,
        "ventana_historial": settings.ventana_historial,
    }


def actual() -> dict:
    return almacen.leer_configuracion(valores_por_defecto())


def guardar(cambios: dict) -> dict:
    almacen.guardar_configuracion(cambios)
    return actual()
