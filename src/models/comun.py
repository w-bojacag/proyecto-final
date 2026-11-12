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
# - umbrales absolutos y la etiqueta.
COLUMNAS_NO_PREDICTORAS = {
    "ANO", "SEMANA", "umbral_brote", "umbral_objetivo", "es_brote_actual", TARGET,
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


def aplicar_umbral_dinamico(pred):
    """Para cada año, el umbral de decisión se calibra con las predicciones fuera de muestra
    del año anterior (información disponible al momento de emitir la alerta)."""
    pred = pred.sort_values(["ANO", "SEMANA"]).copy()
    pred["umbral_decision"] = np.nan
    anos = sorted(pred["ANO"].unique())
    for anterior, actual in zip(anos[:-1], anos[1:]):
        prev = pred[pred["ANO"] == anterior]
        if prev["y"].nunique() < 2:  # año sin ambas clases: se usan todos los años previos
            prev = pred[pred["ANO"] <= anterior]
        pred.loc[pred["ANO"] == actual, "umbral_decision"] = umbral_youden(prev["y"], prev["prob"])
    pred["alerta"] = (pred["prob"] >= pred["umbral_decision"]).astype(int)
    return pred.dropna(subset=["umbral_decision"])


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def resumir_predicciones(pred, particion):
    """Métricas de un modelo a partir de sus predicciones fuera de muestra (ANO, SEMANA, y, prob)."""
    pred = aplicar_umbral_dinamico(pred)
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
    }


def carpeta_walk_forward(rutas, modelo):
    """Modelos de cada año del walk-forward: son los que producen las métricas de validación y
    prueba, y permiten reproducirlas cargando el modelo del año correspondiente."""
    carpeta = rutas.modelos / "walk_forward" / modelo
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta
