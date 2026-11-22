"""API del prototipo de alerta temprana de dengue (Bolívar).

Ejecutar desde src_api/:  uvicorn app.main:app --reload
Documentación interactiva: http://localhost:8000/docs

Las series se cargan en SQLite desde caracteristicas.csv (POST /api/v1/datos/{municipio_id}/csv).
El modelo es la regresión logística del paquete dengue-modelo-logistica (.whl).
"""
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from . import ajustes, almacen
from .config import settings
from .datos import MUNICIPIOS, rango_de_corte, rango_fechas_corte, serie_municipio
from .metricas import PERIODO_PRUEBA, cargar_metricas
from .modelo import cargar_predictor
from .schemas import (
    ConfiguracionIn,
    ConfiguracionOut,
    EstadoDatosOut,
    EstadoMunicipioOut,
    MetricasMunicipioOut,
    ModeloInfoOut,
    MunicipioOut,
    PrediccionIn,
    PrediccionOut,
    PuntoSerieOut,
    RangoFechasOut,
    ResultadoCargaOut,
)

app = FastAPI(
    title="Alerta temprana de dengue – Bolívar (prototipo)",
    version=settings.model_version,
    description="Apoyo a la decisión. No reemplaza el criterio epidemiológico.",
)

predictor = cargar_predictor()
METRICAS = cargar_metricas()


def _sin_datos(nombre: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"No hay datos cargados para {nombre}. Cárguelos en la pestaña Configuración.",
    )


@app.get("/api/v1/health")
def health():
    return {
        "estado": "ok",
        "modelo": predictor.nombre,
        "version": settings.model_version,
    }


@app.get("/api/v1/municipios", response_model=list[MunicipioOut])
def listar_municipios():
    return [MunicipioOut(id=k, nombre=v) for k, v in MUNICIPIOS.items()]


@app.get("/api/v1/configuracion", response_model=ConfiguracionOut)
def leer_configuracion():
    return ConfiguracionOut(**ajustes.actual())


@app.put("/api/v1/configuracion", response_model=ConfiguracionOut)
def guardar_configuracion(body: ConfiguracionIn):
    """Guarda los parámetros indicados en SQLite. Los campos omitidos no cambian."""
    cambios = body.model_dump(exclude_none=True)
    return ConfiguracionOut(**ajustes.guardar(cambios))


@app.get("/api/v1/rango-fechas", response_model=RangoFechasOut)
def rango_fechas():
    rango = rango_fechas_corte(ajustes.actual()["ventana_historial"])
    if rango is None:
        raise HTTPException(status_code=404, detail="No hay datos cargados en SQLite.")
    inicio, fin = rango
    return RangoFechasOut(
        fecha_corte_min=inicio.date(),
        fecha_corte_max=fin.date(),
        horizonte_semanas=settings.horizonte_semanas,
    )


@app.get("/api/v1/municipios/{municipio_id}/serie", response_model=list[PuntoSerieOut])
def serie(municipio_id: str):
    if municipio_id not in MUNICIPIOS:
        raise HTTPException(status_code=404, detail=f"Municipio desconocido: {municipio_id}")
    df = serie_municipio(municipio_id)
    if df is None:
        raise _sin_datos(MUNICIPIOS[municipio_id])
    # Gráficas: se omiten los años anteriores al año mínimo configurado (la predicción usa la serie completa).
    df = df[df["anio"] >= ajustes.actual()["anio_minimo_graficos"]]
    return [
        PuntoSerieOut(
            semana=r.semana.date(),
            anio=int(r.anio),
            mes=int(r.mes),
            casos=int(r.casos),
            lluvia_mm=float(r.lluvia_mm),
            temp_media_c=float(r.temp_media_c),
            umbral_canal=float(r.umbral_canal),
            exceso=bool(r.exceso),
            etiqueta=None if pd.isna(r.etiqueta) else int(r.etiqueta),
        )
        for r in df.itertuples()
    ]


@app.post("/api/v1/prediccion", response_model=PrediccionOut)
def prediccion(body: PrediccionIn):
    if body.municipio_id not in MUNICIPIOS:
        raise HTTPException(status_code=404, detail=f"Municipio desconocido: {body.municipio_id}")
    df = serie_municipio(body.municipio_id)
    if df is None:
        raise _sin_datos(MUNICIPIOS[body.municipio_id])

    cfg = ajustes.actual()
    inicio, fin = rango_de_corte(df["semana"].min(), df["semana"].max(), cfg["ventana_historial"])

    # Se usa la semana epidemiológica (inicia en domingo) que contiene la fecha indicada.
    previas = df.loc[df["semana"] <= pd.Timestamp(body.fecha_corte), "semana"]
    if previas.empty:
        raise HTTPException(status_code=422, detail="fecha_corte anterior a los datos disponibles.")
    fecha = previas.max()
    if not (inicio <= fecha <= fin):
        raise HTTPException(
            status_code=422,
            detail=f"fecha_corte fuera del rango con datos: {inicio.date()} a {fin.date()}.",
        )

    historial = df[df["semana"] <= fecha]  # el modelo sólo ve el pasado
    objetivo_fecha = fecha + pd.Timedelta(weeks=settings.horizonte_semanas)
    filas_objetivo = df[df["semana"] == objetivo_fecha]
    if filas_objetivo.empty:
        raise HTTPException(status_code=422, detail=f"Falta la semana objetivo {objetivo_fecha.date()} en los datos.")
    fila_objetivo = filas_objetivo.iloc[0]

    pred = predictor.predecir(historial, body.municipio_id, fecha, float(fila_objetivo["umbral_canal"]))

    return PrediccionOut(
        municipio_id=body.municipio_id,
        municipio=MUNICIPIOS[body.municipio_id],
        fecha_corte=fecha.date(),
        semana_objetivo=objetivo_fecha.date(),
        horizonte_semanas=settings.horizonte_semanas,
        probabilidad_exceso=pred.probabilidad_exceso,
        alerta=pred.probabilidad_exceso >= cfg["umbral_decision"],
        umbral_decision=cfg["umbral_decision"],
        umbral_canal_objetivo=float(fila_objetivo["umbral_canal"]),
        exceso_observado=bool(fila_objetivo["exceso"]),
        variables_influyentes=pred.variables_influyentes,
        modelo=predictor.nombre,
        version_modelo=settings.model_version,
    )


@app.get("/api/v1/modelo", response_model=ModeloInfoOut)
def modelo_info():
    return ModeloInfoOut(
        nombre=predictor.nombre,
        version=settings.model_version,
        horizonte_semanas=settings.horizonte_semanas,
        umbral_decision=ajustes.actual()["umbral_decision"],
        variables=predictor.variables,
        periodo_prueba=PERIODO_PRUEBA,
        metricas=[MetricasMunicipioOut(**m) for m in METRICAS],
    )


@app.get("/api/v1/datos/estado", response_model=EstadoDatosOut)
def estado_datos():
    """Por municipio, lo que hay guardado en SQLite."""
    return EstadoDatosOut(
        municipios=[
            EstadoMunicipioOut(id=k, nombre=v, **almacen.estado_municipio(k))
            for k, v in MUNICIPIOS.items()
        ],
    )


@app.post("/api/v1/datos/{municipio_id}/csv", response_model=ResultadoCargaOut)
async def cargar_csv(municipio_id: str, archivo: UploadFile = File(...)):
    """Carga caracteristicas.csv (semanal) en SQLite.

    Las semanas que ya existen se actualizan; las nuevas se agregan; las demás no se tocan.
    """
    if municipio_id not in MUNICIPIOS:
        raise HTTPException(status_code=404, detail=f"Municipio desconocido: {municipio_id}")
    contenido = await archivo.read()
    try:
        df = almacen.leer_csv(contenido)
    except almacen.CsvInvalido as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    resultado = almacen.guardar_csv(municipio_id, archivo.filename or "sin_nombre.csv", df)
    return ResultadoCargaOut(id=municipio_id, nombre=MUNICIPIOS[municipio_id], **resultado)
