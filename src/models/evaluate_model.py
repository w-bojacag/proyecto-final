"""
Evaluación clínica comparativa de los modelos de alerta temprana (brote a T+4, Cartagena).

Compara XGBoost, LSTM y regresión logística con dos líneas base que no se entrenan:
- Persistencia: si hay brote en t, se anuncia brote en t+4.
- Regla casos/umbral: casos de la semana t divididos por el umbral de brote de la semana t+4.

Todas las predicciones son fuera de muestra (walk-forward). El umbral de decisión de cada año se
calibra con el índice de Youden sobre las predicciones del año anterior.

Incertidumbre: bootstrap por bloques de semanas consecutivas sobre el periodo de prueba
(las semanas vecinas están correlacionadas, así que no se remuestrean de forma independiente).
Para comparar modelos se usa la diferencia de AUC en las mismas muestras bootstrap.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comun import ROOT_DIR, TARGET, aplicar_umbral_dinamico, cargar_matriz, load_params, resumir_predicciones

REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "eval_metrics.json"
MODELOS = {
    "xgboost": REPORTS_DIR / "predicciones_xgboost.csv",
    "lstm": REPORTS_DIR / "predicciones_lstm.csv",
    "logistica": REPORTS_DIR / "predicciones_logistica.csv",
}
REFERENCIA = "logistica"  # modelo contra el que se comparan los demás


def remuestras_por_bloques(n, tamano_bloque, repeticiones, semilla):
    """Índices bootstrap formados por bloques consecutivos (circulares) de semanas."""
    rng = np.random.default_rng(semilla)
    n_bloques = int(np.ceil(n / tamano_bloque))
    for _ in range(repeticiones):
        inicios = rng.integers(0, n, n_bloques)
        idx = (inicios[:, None] + np.arange(tamano_bloque)[None, :]).ravel() % n
        yield idx[:n]


def bootstrap(predicciones, particion, params):
    """Intervalos del 95 % para el AUC y la sensibilidad de prueba, y diferencias de AUC
    de cada modelo frente al de referencia."""
    prueba = {}
    for nombre, p in predicciones.items():
        p = aplicar_umbral_dinamico(p)
        prueba[nombre] = p[p["ANO"] > particion["val_end_year"]].reset_index(drop=True)
    y = prueba[REFERENCIA]["y"].to_numpy()

    aucs = {n: [] for n in prueba}
    sens = {n: [] for n in prueba}
    for idx in remuestras_por_bloques(len(y), params["tamano_bloque"], params["repeticiones"], params["semilla"]):
        if len(np.unique(y[idx])) < 2:
            continue
        for nombre, p in prueba.items():
            aucs[nombre].append(roc_auc_score(y[idx], p["prob"].to_numpy()[idx]))
            positivos = y[idx] == 1
            sens[nombre].append(p["alerta"].to_numpy()[idx][positivos].mean())

    def ic(valores):
        return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]

    resultado = {}
    for nombre in prueba:
        diferencia = np.array(aucs[nombre]) - np.array(aucs[REFERENCIA])
        resultado[nombre] = {
            "ic95_roc_auc_test": ic(aucs[nombre]),
            "ic95_sensibilidad_test": ic(sens[nombre]),
            f"dif_auc_vs_{REFERENCIA}": float(np.mean(diferencia)),
            f"ic95_dif_auc_vs_{REFERENCIA}": ic(diferencia),
        }
    return resultado


def evaluar_modelos():
    particion = load_params("particion")
    params = load_params("evaluate")
    df = cargar_matriz()
    df = df[df["ANO"] >= particion["train_end_year"]]
    base = df[["ANO", "SEMANA"]].assign(y=df[TARGET].astype(int))

    predicciones = {
        nombre: pd.read_csv(ruta).sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
        for nombre, ruta in MODELOS.items()
    }
    predicciones["base_persistencia"] = base.assign(prob=df["es_brote_actual"].astype(float))
    predicciones["base_casos_sobre_umbral"] = base.assign(prob=df["ratio_casos_umbral_objetivo"])
    for nombre, p in predicciones.items():  # todos deben evaluar exactamente las mismas semanas
        if not p[["ANO", "SEMANA"]].reset_index(drop=True).equals(base[["ANO", "SEMANA"]].reset_index(drop=True)):
            raise ValueError(f"Las semanas de '{nombre}' no coinciden con las de la matriz")

    resultados = {nombre: resumir_predicciones(p, particion) for nombre, p in predicciones.items()}
    for nombre, extra in bootstrap(predicciones, particion, params).items():
        resultados[nombre].update(extra)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)

    filas = []
    for nombre, r in resultados.items():
        filas.append({
            "modelo": nombre,
            "AUC val": f"{r['roc_auc_val']:.3f}",
            "AUC prueba [IC95]": f"{r['roc_auc_test']:.3f} [{r['ic95_roc_auc_test'][0]:.2f}-{r['ic95_roc_auc_test'][1]:.2f}]",
            f"dif. vs {REFERENCIA} [IC95]": (
                f"{r[f'dif_auc_vs_{REFERENCIA}']:+.3f} "
                f"[{r[f'ic95_dif_auc_vs_{REFERENCIA}'][0]:+.2f} a {r[f'ic95_dif_auc_vs_{REFERENCIA}'][1]:+.2f}]"
            ),
            "PR-AUC": f"{r['pr_auc_test']:.3f}",
            "sensib. [IC95]": f"{r['test_sensibilidad']:.0%} [{r['ic95_sensibilidad_test'][0]:.0%}-{r['ic95_sensibilidad_test'][1]:.0%}]",
            "especif.": f"{r['test_especificidad']:.0%}",
            "precisión": f"{r['test_precision']:.0%}",
            "VP/FP/FN/VN": f"{r['test_vp']}/{r['test_fp']}/{r['test_fn']}/{r['test_vn']}",
        })
    print("=" * 70)
    print("EVALUACIÓN COMPARATIVA (walk-forward; prueba 2023-2025)")
    print("=" * 70)
    print(f"Prevalencia de brote en prueba: {resultados[REFERENCIA]['prevalencia_test']:.1%}")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nMétricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    evaluar_modelos()
