"""
Evaluación clínica comparativa de los modelos de alerta temprana (brote a T+4, Cartagena).

Compara XGBoost y LSTM con dos líneas base que no se entrenan:
- Persistencia: si hay brote en t, se anuncia brote en t+4.
- Regla casos/umbral: casos de la semana t divididos por el umbral de brote de la semana t+4.
Un modelo solo aporta valor si supera a estas reglas.

Todas las predicciones son fuera de muestra (walk-forward). El umbral de decisión de cada año se
calibra con el índice de Youden sobre las predicciones del año anterior, de modo que la matriz
de confusión de prueba no usa información de los años evaluados.
"""

import json

import pandas as pd

from comun import ROOT_DIR, TARGET, cargar_matriz, load_params, resumir_predicciones

REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "eval_metrics.json"
MODELOS = {
    "xgboost": REPORTS_DIR / "predicciones_xgboost.csv",
    "lstm": REPORTS_DIR / "predicciones_lstm.csv",
}


def evaluar_modelos():
    particion = load_params("particion")
    df = cargar_matriz()
    df = df[df["ANO"] >= particion["train_end_year"]]
    base = df[["ANO", "SEMANA"]].assign(y=df[TARGET].astype(int))

    predicciones = {nombre: pd.read_csv(ruta) for nombre, ruta in MODELOS.items()}
    predicciones["base_persistencia"] = base.assign(prob=df["es_brote_actual"].astype(float))
    predicciones["base_casos_sobre_umbral"] = base.assign(prob=df["ratio_casos_umbral_objetivo"])

    resultados = {nombre: resumir_predicciones(p, particion) for nombre, p in predicciones.items()}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)

    tabla = pd.DataFrame(resultados).T[[
        "roc_auc_val", "roc_auc_test", "pr_auc_test", "test_sensibilidad", "test_especificidad",
        "test_precision", "test_vp", "test_fp", "test_fn", "test_vn",
    ]]
    print("=" * 70)
    print("EVALUACIÓN COMPARATIVA (walk-forward; prueba 2023-2025)")
    print("=" * 70)
    print(f"Prevalencia de brote en prueba: {resultados['xgboost']['prevalencia_test']:.1%}")
    print(tabla.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nMétricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    evaluar_modelos()
