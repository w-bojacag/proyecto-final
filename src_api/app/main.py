"""API del prototipo de alerta temprana de dengue (Bolívar).

Ejecutar desde src_api/:  uvicorn app.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import settings
from .datos import MUNICIPIOS, rango_fechas_corte, serie_municipio
from .modelo import cargar_predictor
from .schemas import (
    ModeloInfoOut,
    MunicipioOut,
    PrediccionIn,
    PrediccionOut,
    PuntoSerieOut,
    RangoFechasOut,
)

ADVERTENCIA_SIMULADO = (
    "PROTOTIPO: datos y modelo simulados. Resultado ilustrativo, no es una predicción validada."
)

app = FastAPI(
    title="Alerta temprana de dengue – Bolívar (prototipo)",
    version=settings.model_version,
    description="Apoyo a la decisión. No reemplaza el criterio epidemiológico.",
)

predictor = cargar_predictor()


@app.get("/api/v1/health")
def health():
    return {"estado": "ok", "backend": predictor.backend, "version": settings.model_version}


@app.get("/api/v1/municipios", response_model=list[MunicipioOut])
def listar_municipios():
    return [MunicipioOut(id=k, nombre=v) for k, v in MUNICIPIOS.items()]


@app.get("/api/v1/rango-fechas", response_model=RangoFechasOut)
def rango_fechas():
    inicio, fin = rango_fechas_corte()
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
    return [
        PuntoSerieOut(
            semana=r.semana.date(),
            casos=int(r.casos),
            lluvia_mm=float(r.lluvia_mm),
            temp_media_c=float(r.temp_media_c),
            umbral_canal=float(r.umbral_canal),
            exceso=bool(r.exceso),
        )
        for r in df.itertuples()
    ]


@app.post("/api/v1/prediccion", response_model=PrediccionOut)
def prediccion(body: PrediccionIn):
    if body.municipio_id not in MUNICIPIOS:
        raise HTTPException(status_code=404, detail=f"Municipio desconocido: {body.municipio_id}")

    df = serie_municipio(body.municipio_id)
    inicio, fin = rango_fechas_corte()

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
    fila_objetivo = df[df["semana"] == objetivo_fecha].iloc[0]

    pred = predictor.predecir(historial, body.municipio_id, fecha)

    return PrediccionOut(
        municipio_id=body.municipio_id,
        municipio=MUNICIPIOS[body.municipio_id],
        fecha_corte=fecha.date(),
        semana_objetivo=objetivo_fecha.date(),
        horizonte_semanas=settings.horizonte_semanas,
        probabilidad_exceso=pred.probabilidad_exceso,
        alerta=pred.probabilidad_exceso >= settings.umbral_decision,
        umbral_decision=settings.umbral_decision,
        umbral_canal_objetivo=float(fila_objetivo["umbral_canal"]),
        exceso_observado=bool(fila_objetivo["exceso"]),
        variables_influyentes=pred.variables_influyentes,
        modelo=predictor.nombre,
        backend=predictor.backend,
        version_modelo=settings.model_version,
        advertencia=ADVERTENCIA_SIMULADO if predictor.backend == "simulado" else "",
    )


@app.get("/api/v1/modelo", response_model=ModeloInfoOut)
def modelo_info():
    return ModeloInfoOut(
        nombre=predictor.nombre,
        backend=predictor.backend,
        version=settings.model_version,
        horizonte_semanas=settings.horizonte_semanas,
        umbral_decision=settings.umbral_decision,
        variables=predictor.variables,
        metricas="PENDIENTE: recall, especificidad, precisión, F1 y AUC-ROC por municipio aún no calculados.",
        advertencia=ADVERTENCIA_SIMULADO if predictor.backend == "simulado" else "",
    )
