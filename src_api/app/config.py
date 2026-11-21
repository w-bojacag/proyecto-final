"""Configuración externalizada por variables de entorno (ver README)."""
import os
from dataclasses import dataclass
from pathlib import Path

RAIZ_API = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # "demo": series simuladas (por defecto). "sqlite": series semanales cargadas desde caracteristicas.csv.
    modo_datos: str = os.getenv("MODO_DATOS", "demo")
    sqlite_path: str = os.getenv("SQLITE_PATH", str(RAIZ_API / "data" / "dengue.sqlite"))
    # "simulado" mientras no exista el modelo entrenado. Cambiar al backend real
    # cuando se implemente (ver app/modelo.py: cargar_predictor).
    model_backend: str = os.getenv("MODEL_BACKEND", "simulado")
    model_version: str = os.getenv("MODEL_VERSION", "simulado-0.1")
    horizonte_semanas: int = int(os.getenv("HORIZONTE_SEMANAS", "4"))
    # Umbral de probabilidad para emitir alerta. Pendiente de validar con el equipo.
    umbral_decision: float = float(os.getenv("UMBRAL_DECISION", "0.5"))
    anio_inicio: int = int(os.getenv("ANIO_INICIO", "2007"))
    anio_fin: int = int(os.getenv("ANIO_FIN", "2025"))
    # Partición temporal de la propuesta: entrenamiento 2007-2019.
    anio_entrenamiento_fin: int = int(os.getenv("ANIO_ENTRENAMIENTO_FIN", "2019"))
    # Primer año que se muestra en las gráficas (canal endémico y clima, endpoint /serie).
    # Los primeros años tienen umbrales calculados con poca historia; la predicción sigue usando toda la serie.
    anio_minimo_graficos: int = int(os.getenv("ANIO_MINIMO_GRAFICOS", "2012"))
    semilla: int = int(os.getenv("SEMILLA_SIMULACION", "42"))
    # Semanas de historial mínimo que requiere el modelo.
    ventana_historial: int = int(os.getenv("VENTANA_HISTORIAL", "8"))


settings = Settings()
