"""Métricas de evaluación de la regresión logística por municipio (conjunto de prueba 2023–2025).

Fuente: `metricas/logistica_por_municipio.csv`, copia de las filas `logistica` de
`reports/resumen_municipios.csv` generado por el pipeline. Para actualizarla tras reentrenar,
volver a copiar esas filas (ver src_api/README.md, sección «Actualizar las métricas»).
"""
from pathlib import Path

import pandas as pd

RUTA_METRICAS = Path(__file__).resolve().parent.parent / "metricas" / "logistica_por_municipio.csv"
PERIODO_PRUEBA = "2023–2025"


def _f1(precision: float, recall: float) -> float | None:
    if precision is None or recall is None or pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def cargar_metricas() -> list[dict]:
    """Métricas por municipio. Lista vacía si el archivo no existe."""
    if not RUTA_METRICAS.exists():
        return []
    df = pd.read_csv(RUTA_METRICAS)
    salida = []
    for r in df.itertuples():
        precision = float(r.test_precision)
        recall = float(r.test_sensibilidad)
        salida.append({
            "municipio_id": r.municipio,
            "auc_roc": float(r.roc_auc_test),
            "auc_pr": float(r.pr_auc_test),
            "recall": recall,
            "especificidad": float(r.test_especificidad),
            "precision": precision,
            "f1": _f1(precision, recall),
            "auc_roc_ic95": [float(r.ic95_auc_inf), float(r.ic95_auc_sup)],
            "vp": int(r.test_vp),
            "fp": int(r.test_fp),
            "fn": int(r.test_fn),
            "vn": int(r.test_vn),
            "brotes_prueba": int(r.test_brotes),
            "brotes_avisados": int(r.test_brotes_avisados),
        })
    return salida
