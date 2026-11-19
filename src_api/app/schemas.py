from datetime import date

from pydantic import BaseModel, Field


class MunicipioOut(BaseModel):
    id: str
    nombre: str


class RangoFechasOut(BaseModel):
    fecha_corte_min: date
    fecha_corte_max: date
    horizonte_semanas: int


class PuntoSerieOut(BaseModel):
    semana: date
    casos: int
    lluvia_mm: float
    temp_media_c: float
    umbral_canal: float
    exceso: bool


class PrediccionIn(BaseModel):
    municipio_id: str = Field(..., examples=["cartagena"])
    fecha_corte: date = Field(..., description="Última semana con datos conocidos (inicio de semana).")


class PrediccionOut(BaseModel):
    municipio_id: str
    municipio: str
    fecha_corte: date
    semana_objetivo: date
    horizonte_semanas: int
    probabilidad_exceso: float
    alerta: bool
    umbral_decision: float
    umbral_canal_objetivo: float
    exceso_observado: bool | None = Field(
        None, description="Valor registrado en la semana objetivo; sirve sólo para comparar."
    )
    variables_influyentes: list[dict]
    modelo: str
    backend: str
    version_modelo: str
    advertencia: str


class ModeloInfoOut(BaseModel):
    nombre: str
    backend: str
    version: str
    horizonte_semanas: int
    umbral_decision: float
    variables: list[str]
    metricas: str
    advertencia: str
