"""
Utilidades compartidas por los modelos de alerta temprana (XGBoost y LSTM).

Ambos modelos usan la misma etiqueta, el mismo protocolo de evaluación (walk-forward anual)
y las mismas métricas, para que sus resultados sean comparables entre sí.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rutas import ROOT_DIR, cargar_params, municipio_desde_argumentos  # noqa: E402,F401

TARGET = "target_brote_t4"

# Columnas que nunca entran al modelo:
# - identificadores de tiempo;
# - conteos de la semana t en valor absoluto (se usan en forma relativa al umbral, porque su
#   escala cambió mucho entre la fase endémica y la hiperendémica);
# - umbrales absolutos, la línea central del canal (solo la usan las líneas base) y la etiqueta.
COLUMNAS_NO_PREDICTORAS = {
    "ANO", "SEMANA", "umbral_brote", "umbral_objetivo", "centro_canal", "es_brote_actual", TARGET,
}


def load_params(seccion=None):
    return cargar_params(seccion)


def cargar_matriz(rutas):
    df = pd.read_csv(rutas.caracteristicas)
    df = df.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def columnas_predictoras(df):
    casos_semana_t = [
        c for c in df.columns
        if c.startswith("casos_") and not any(s in c for s in ("_lag_", "_roll_", "_tasa_"))
    ]
    return [c for c in df.columns if c not in COLUMNAS_NO_PREDICTORAS and c not in casos_semana_t]


def anos_walk_forward(df, particion):
    """Años que se predicen fuera de muestra: desde el último de entrenamiento hasta el final.

    El primer año (train_end_year) solo sirve para calibrar el umbral de decisión del siguiente.
    """
    return [a for a in sorted(df["ANO"].unique()) if a >= particion["train_end_year"]]


def mascara_entrenamiento(anos, ano_prediccion, horizonte):
    """Semanas para entrenar el modelo que predice `ano_prediccion`: todas las anteriores,
    menos las últimas `horizonte`, cuya etiqueta (brote en t+h) cae dentro del año a predecir
    y todavía no se conocería al emitir las alertas de ese año."""
    mascara = np.asarray(anos) < ano_prediccion
    posiciones = np.where(mascara)[0]
    mascara[posiciones[-horizonte:]] = False
    return mascara


def umbral_youden(y, p):
    """Umbral que maximiza J = sensibilidad - tasa de falsos positivos."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    umbral = thr[np.argmax(tpr - fpr)]
    # roc_curve antepone un umbral infinito (ninguna alerta); se reemplaza por el máximo observado
    return float(umbral if np.isfinite(umbral) else np.max(p) + 1e-9)


def aplicar_umbral_dinamico(pred, umbral_fijo=None):
    """Para cada año, el umbral de decisión se calibra con las predicciones fuera de muestra
    del año anterior (información disponible al momento de emitir la alerta).

    `umbral_fijo` sirve para reglas que ya emiten alertas binarias (la alerta del canal
    endémico): se usa ese umbral en lugar de Youden, sobre las mismas semanas."""
    pred = pred.sort_values(["ANO", "SEMANA"]).copy()
    pred["umbral_decision"] = np.nan
    anos = sorted(pred["ANO"].unique())
    for anterior, actual in zip(anos[:-1], anos[1:]):
        prev = pred[pred["ANO"] == anterior]
        if prev["y"].nunique() < 2:  # año sin ambas clases: se usan todos los años previos
            prev = pred[pred["ANO"] <= anterior]
        umbral = umbral_fijo if umbral_fijo is not None else umbral_youden(prev["y"], prev["prob"])
        pred.loc[pred["ANO"] == actual, "umbral_decision"] = umbral
    pred["alerta"] = (pred["prob"] >= pred["umbral_decision"]).astype(int)
    return pred.dropna(subset=["umbral_decision"])


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def resumir_predicciones(pred, particion, umbral_fijo=None):
    """Métricas de un modelo a partir de sus predicciones fuera de muestra (ANO, SEMANA, y, prob)."""
    pred = aplicar_umbral_dinamico(pred, umbral_fijo)
    val = pred[pred["ANO"] <= particion["val_end_year"]]
    test = pred[pred["ANO"] > particion["val_end_year"]]
    tn, fp, fn, tp = confusion_matrix(test["y"], test["alerta"], labels=[0, 1]).ravel()
    return {
        "roc_auc_val": _auc(val["y"], val["prob"]),
        "roc_auc_test": _auc(test["y"], test["prob"]),
        "pr_auc_test": float(average_precision_score(test["y"], test["prob"])),
        "prevalencia_test": float(test["y"].mean()),
        "roc_auc_por_ano": {int(a): _auc(g["y"], g["prob"]) for a, g in pred.groupby("ANO")},
        "test_sensibilidad": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "test_especificidad": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "test_precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "test_vp": int(tp), "test_fp": int(fp), "test_fn": int(fn), "test_vn": int(tn),
        **metricas_evento(test, cargar_params("target")["horizonte"], **cargar_params("eventos")),
    }


def episodios(serie, separacion_minima):
    """Tramos de semanas consecutivas con valor 1, como pares (inicio, fin) de posiciones.
    Dos tramos separados por menos de `separacion_minima` semanas en 0 se consideran uno solo
    (un nuevo brote exige al menos 2 semanas previas por debajo del umbral)."""
    tramos, inicio, ceros = [], None, 0
    for i, v in enumerate(serie):
        if v == 1:
            if inicio is None:
                inicio = i
            elif ceros:
                if ceros < separacion_minima:
                    pass  # pausa corta: sigue el mismo episodio
                else:
                    tramos.append((inicio, i - ceros - 1))
                    inicio = i
            ceros = 0
        elif inicio is not None:
            ceros += 1
    if inicio is not None:
        tramos.append((inicio, len(serie) - 1 - ceros))
    return tramos


def metricas_evento(pred, horizonte, separacion_minima, anticipacion):
    """Métricas por evento de Umaña et al. (2026), adaptadas a una alerta que se emite h semanas
    antes de la semana objetivo. Las series se leen en tiempo de la semana objetivo (t+h).

    - Brote: episodio de semanas con brote real. Los que ya están en curso en la primera semana
      evaluada no cuentan, porque su inicio no se observa.
    - Brote avisado: hay alguna alerta para su semana de inicio o para las `anticipacion`
      semanas anteriores.
    - Capacidad de detección = brotes avisados / brotes.
    - Razón de alertas = episodios de alerta que anteceden o coinciden con un brote / episodios
      de alerta emitidos.
    - Desempeño = promedio de las dos (alfa = 0,5, como en el artículo).
    - Ventana de oportunidad: semanas entre la emisión de la primera alerta del episodio que avisó
      el brote y el inicio del brote (incluye las h semanas del horizonte).
    """
    y = pred["y"].to_numpy()
    alerta = pred["alerta"].to_numpy()
    brotes = [b for b in episodios(y, separacion_minima) if b[0] > 0]
    alertas = episodios(alerta, separacion_minima)

    avisados, ventanas = 0, []
    for inicio, _ in brotes:
        previas = [a for a in alertas if a[0] <= inicio and a[1] >= inicio - anticipacion]
        if previas:
            avisados += 1
            ventanas.append(inicio - previas[0][0] + horizonte)
    utiles = sum(
        any(a[0] <= fin and a[1] >= inicio - anticipacion for inicio, fin in brotes) for a in alertas
    )
    deteccion = avisados / len(brotes) if brotes else float("nan")
    razon = utiles / len(alertas) if alertas else float("nan")
    return {
        "test_brotes": len(brotes),
        "test_brotes_avisados": avisados,
        "test_episodios_alerta": len(alertas),
        "test_capacidad_deteccion": float(deteccion),
        "test_razon_alertas": float(razon),
        "test_desempeno_evento": float(0.5 * deteccion + 0.5 * razon),
        "test_ventana_oportunidad_mediana": float(np.median(ventanas)) if ventanas else float("nan"),
    }


def carpeta_walk_forward(rutas, modelo):
    """Modelos de cada año del walk-forward: son los que producen las métricas de validación y
    prueba, y permiten reproducirlas cargando el modelo del año correspondiente."""
    carpeta = rutas.modelos / "walk_forward" / modelo
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta
