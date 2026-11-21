"""Modelo de predicción de excesos de dengue.

El modelo es la regresión logística del paquete `dengue-modelo-logistica` (.whl en `model_pkg/`,
generado desde `models/paquete_modelo/`). Para agregar otro modelo:
1. Crear una clase que herede de `Predictor`.
2. Construir las variables exactamente como en entrenamiento, usando sólo
   `historial` (datos hasta `fecha_corte`) y `umbral_objetivo`.
3. Devolver la probabilidad de exceso en la semana `fecha_corte + horizonte`.
4. Devolverla desde `cargar_predictor()`.

Los endpoints no cambian.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from .config import settings

# Variables de la regresión logística (params.yaml, sección logistica).
VARIABLES_LOGISTICA = [
    "ratio_casos_umbral_objetivo",
    "ratio_roll4_umbral_objetivo",
    "exceso_actual",
]


@dataclass
class Prediccion:
    probabilidad_exceso: float
    variables_influyentes: list[dict] = field(default_factory=list)


class Predictor(ABC):
    nombre: str
    horizonte: int
    variables: list[str]

    @abstractmethod
    def predecir(
        self,
        historial: pd.DataFrame,
        municipio_id: str,
        fecha_corte: pd.Timestamp,
        umbral_objetivo: float,
    ) -> Prediccion:
        """`historial`: filas del municipio con semana <= fecha_corte (sin fuga de información).

        `umbral_objetivo`: umbral del canal de la semana fecha_corte + horizonte. Se conoce en la
        fecha de corte porque sólo depende de años anteriores (igual que en el pipeline).
        """


class PredictorLogistica(Predictor):
    """Regresión logística empaquetada en `dengue-modelo-logistica`, una por municipio.

    Las variables y el modelo vienen del paquete, así que la API y el entrenamiento usan el mismo código.
    """

    nombre = "Regresión logística"
    horizonte = settings.horizonte_semanas
    variables = VARIABLES_LOGISTICA

    def predecir(
        self,
        historial: pd.DataFrame,
        municipio_id: str,
        fecha_corte: pd.Timestamp,
        umbral_objetivo: float,
    ) -> Prediccion:
        import modelo_dengue  # import tardío: el paquete se instala desde el .whl en la imagen

        if municipio_id not in modelo_dengue.municipios_disponibles():
            raise RuntimeError(f"El paquete del modelo no incluye '{municipio_id}'.")
        resultado = modelo_dengue.predecir(historial, municipio_id, umbral_objetivo)
        return Prediccion(
            probabilidad_exceso=round(resultado["probabilidad_exceso"], 4),
            variables_influyentes=[
                {"variable": v, "aporte": round(a, 3)} for v, a in resultado["aportes_logit"].items()
            ],
        )


def cargar_predictor() -> Predictor:
    return PredictorLogistica()
