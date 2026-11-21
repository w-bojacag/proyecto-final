"""Configuración externalizada por variables de entorno (ver README)."""
import os
from dataclasses import dataclass
from pathlib import Path

RAIZ_API = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Base SQLite con las series semanales cargadas desde caracteristicas.csv.
    sqlite_path: str = os.getenv("SQLITE_PATH", str(RAIZ_API / "data" / "dengue.sqlite"))
    # Versión del modelo reportada en cada respuesta.
    model_version: str = os.getenv("MODEL_VERSION", "logistica-0.1")
    horizonte_semanas: int = int(os.getenv("HORIZONTE_SEMANAS", "4"))
    # Umbral de probabilidad para emitir alerta. Pendiente de validar con el equipo.
    umbral_decision: float = float(os.getenv("UMBRAL_DECISION", "0.5"))
    # Primer año que se muestra en las gráficas (canal endémico y clima, endpoint /serie).
    # Los primeros años tienen umbrales calculados con poca historia; la predicción sigue usando toda la serie.
    anio_minimo_graficos: int = int(os.getenv("ANIO_MINIMO_GRAFICOS", "2012"))
    # Semanas de historial mínimo que requiere el modelo.
    ventana_historial: int = int(os.getenv("VENTANA_HISTORIAL", "8"))


settings = Settings()
