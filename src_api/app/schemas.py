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


class MetricasMunicipioOut(BaseModel):
    municipio_id: str
    auc_roc: float = Field(..., description="AUC-ROC en el conjunto de prueba.")
    auc_pr: float = Field(..., description="Área bajo la curva precisión-recall en el conjunto de prueba.")
    recall: float = Field(..., description="Sensibilidad: proporción de brotes detectados.")
    especificidad: float
    precision: float
    f1: float | None
    auc_roc_ic95: list[float] = Field(..., description="Intervalo del 95 % de la AUC-ROC (inferior, superior).")
    vp: int
    fp: int
    fn: int
    vn: int
    brotes_prueba: int = Field(..., description="Semanas con brote en el conjunto de prueba.")
    brotes_avisados: int = Field(..., description="Brotes con alerta previa dentro de la ventana de anticipación.")


class ConfiguracionIn(BaseModel):
    umbral_decision: float | None = Field(None, ge=0, le=1, description="Probabilidad a partir de la cual hay alerta.")
    anio_minimo_graficos: int | None = Field(None, ge=2007, le=2030, description="Primer año de las gráficas.")
    ventana_historial: int | None = Field(None, ge=1, le=52, description="Semanas de historial mínimo para una fecha de corte.")


class ConfiguracionOut(BaseModel):
    umbral_decision: float
    anio_minimo_graficos: int
    ventana_historial: int


class ModeloInfoOut(BaseModel):
    nombre: str
    version: str
    horizonte_semanas: int
    umbral_decision: float
    variables: list[str]
    periodo_prueba: str
    metricas: list[MetricasMunicipioOut]
