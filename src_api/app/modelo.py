"""Interfaz del modelo de predicción de excesos de dengue.

La API sólo conoce esta interfaz. Para conectar el modelo real:
1. Crear una clase que herede de `Predictor` y cargue el artefacto guardado
   (con su preprocesamiento, según la regla del proyecto).
2. Construir las variables exactamente como en entrenamiento, usando sólo
   `historial` (datos hasta `fecha_corte`).
3. Devolver la probabilidad de exceso en la semana `fecha_corte + horizonte`.
4. Registrarla en `cargar_predictor()`.

Los endpoints no cambian.
"""
import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from .config import settings


@dataclass
class Prediccion:
    probabilidad_exceso: float
    variables_influyentes: list[dict] = field(default_factory=list)


class Predictor(ABC):
    nombre: str
    backend: str
    horizonte: int
    variables: list[str]

    @abstractmethod
    def predecir(
        self, historial: pd.DataFrame, municipio_id: str, fecha_corte: pd.Timestamp
    ) -> Prediccion:
        """`historial`: filas del municipio con semana <= fecha_corte (sin fuga de información)."""


def _ruido_determinista(municipio_id: str, fecha_corte: pd.Timestamp) -> float:
    """Valor estable en [-0.5, 0.5] para que la misma consulta devuelva la misma respuesta."""
    digest = hashlib.sha256(f"{municipio_id}|{fecha_corte.date()}|{settings.semilla}".encode())
    return int(digest.hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5


class PredictorSimulado(Predictor):
    """Placeholder: no es un modelo entrenado. Sirve para probar el flujo completo.

    Los coeficientes son arbitrarios. No deben usarse para afirmar resultados.
    """

    nombre = "Modelo"
    backend = "simulado"
    horizonte = settings.horizonte_semanas
    variables = ["casos_recientes_vs_umbral", "lluvia_reciente_mm", "ruido"]

    def predecir(
        self, historial: pd.DataFrame, municipio_id: str, fecha_corte: pd.Timestamp
    ) -> Prediccion:
        reciente = historial.tail(settings.ventana_historial)
        umbral = max(float(reciente["umbral_canal"].iloc[-1]), 1.0)
        razon = float(reciente["casos"].mean()) / umbral
        lluvia = float(reciente["lluvia_mm"].tail(4).mean())
        ruido = _ruido_determinista(municipio_id, fecha_corte)

        aporte_casos = 1.5 * (razon - 1.0)
        aporte_lluvia = 0.01 * (lluvia - 60.0)
        aporte_ruido = 0.4 * ruido
        z = -0.8 + aporte_casos + aporte_lluvia + aporte_ruido
        prob = 1.0 / (1.0 + math.exp(-z))

        return Prediccion(
            probabilidad_exceso=round(prob, 4),
            variables_influyentes=[
                {"variable": "casos_recientes_vs_umbral", "aporte": round(aporte_casos, 3)},
                {"variable": "lluvia_reciente_mm", "aporte": round(aporte_lluvia, 3)},
                {"variable": "ruido", "aporte": round(aporte_ruido, 3)},
            ],
        )


def cargar_predictor() -> Predictor:
    if settings.model_backend == "simulado":
        return PredictorSimulado()
    raise RuntimeError(
        f"Backend de modelo no implementado: '{settings.model_backend}'. "
        "Agregar la clase en app/modelo.py y registrarla aquí."
    )
