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
    semana: date = Field(..., description="Domingo que inicia la semana epidemiológica.")
    anio: int
    mes: int = Field(..., description="Mes del domingo de inicio de la semana.")
    casos: int
    lluvia_mm: float
    temp_media_c: float
    umbral_canal: float
    exceso: bool
    etiqueta: int | None = Field(
        None, description="Brote en la semana actual (0 o 1), según es_brote_actual del pipeline. Vacío si no se cargó."
    )


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
    version_modelo: str


class EstadoMunicipioOut(BaseModel):
    id: str
    nombre: str
    cargado: bool
    semanas: int = Field(..., description="Semanas guardadas en SQLite para el municipio.")
    primer_periodo: str | None = Field(None, examples=["2010-S01"])
    ultimo_periodo: str | None = Field(None, examples=["2025-S49"])
    archivo: str | None
    cargado_en: str | None
    ultimas_nuevas: int = Field(0, description="Semanas agregadas en la última carga.")
    ultimas_actualizadas: int = Field(0, description="Semanas que ya existían y se actualizaron en la última carga.")


class ResultadoCargaOut(EstadoMunicipioOut):
    nuevas: int = Field(..., description="Semanas nuevas en esta carga.")
    actualizadas: int = Field(..., description="Semanas existentes que se actualizaron en esta carga.")


class EstadoDatosOut(BaseModel):
    municipios: list[EstadoMunicipioOut]


class ModeloInfoOut(BaseModel):
    nombre: str
    version: str
    horizonte_semanas: int
    umbral_decision: float
    variables: list[str]
    metricas: str
