"""Modelo de regresión logística para alerta temprana de excesos de dengue (Bolívar).

Uso:
    from modelo_dengue import predecir
    resultado = predecir(historial, "cartagena", umbral_objetivo)

`historial`: DataFrame con las columnas `casos` y `umbral_canal`, una fila por semana,
en orden cronológico y terminando en la fecha de corte.
`umbral_objetivo`: umbral del canal de la semana fecha de corte + 4 semanas.

Las variables se calculan igual que en el entrenamiento (src/features/build_features.py).
"""
from pathlib import Path

import pandas as pd

VARIABLES = [
    "ratio_casos_umbral_objetivo",
    "ratio_roll4_umbral_objetivo",
    "exceso_actual",
]
HORIZONTE_SEMANAS = 4
_CARPETA_ARTEFACTOS = Path(__file__).resolve().parent / "artefactos"
_cache: dict[str, object] = {}


def municipios_disponibles() -> list[str]:
    return sorted(p.name for p in _CARPETA_ARTEFACTOS.iterdir() if (p / "logistica.joblib").exists())


def cargar(municipio: str):
    """Carga (una vez) el modelo logístico del municipio."""
    if municipio not in _cache:
        import joblib  # import tardío: la carga real requiere scikit-learn

        ruta = _CARPETA_ARTEFACTOS / municipio / "logistica.joblib"
        if not ruta.exists():
            raise FileNotFoundError(f"No hay modelo para '{municipio}'. Disponibles: {municipios_disponibles()}")
        _cache[municipio] = joblib.load(ruta)
    return _cache[municipio]


def construir_variables(historial: pd.DataFrame, umbral_objetivo: float) -> dict[str, float]:
    """Calcula las tres variables de la fecha de corte (última fila del historial)."""
    if len(historial) < 5:
        raise ValueError("Se necesitan al menos 5 semanas de historial.")
    casos = historial["casos"].to_numpy(dtype=float)
    casos_t = casos[-1]
    media_previas = casos[-5:-1].mean()  # semanas t-4 a t-1
    umbral_t = float(historial["umbral_canal"].iloc[-1])
    return {
        "ratio_casos_umbral_objetivo": casos_t / (umbral_objetivo + 1.0),
        "ratio_roll4_umbral_objetivo": media_previas / (umbral_objetivo + 1.0),
        "exceso_actual": casos_t / (umbral_t + 1.0),
    }


def predecir(historial: pd.DataFrame, municipio: str, umbral_objetivo: float) -> dict:
    """Probabilidad de exceso en la semana objetivo y aporte de cada variable (en logit)."""
    modelo = cargar(municipio)
    X = pd.DataFrame([construir_variables(historial, umbral_objetivo)])[VARIABLES]
    probabilidad = float(modelo.predict_proba(X)[0, 1])
    estandarizado = modelo[:-1].transform(X)[0]
    coeficientes = modelo[-1].coef_[0]
    aportes = {v: float(c * z) for v, c, z in zip(VARIABLES, coeficientes, estandarizado)}
    return {"municipio": municipio, "probabilidad_exceso": probabilidad, "aportes_logit": aportes}
