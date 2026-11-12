"""
Entrenamiento de XGBoost para la alerta temprana de brote de dengue a T+4 (Cartagena).

Protocolo walk-forward anual: para predecir el año Y se entrena con todos los años anteriores.
Las predicciones fuera de muestra de validación (2020-2022) y prueba (2023-2025) se guardan en
reports/predicciones_xgboost.csv; evaluate_model.py calcula sobre ellas las métricas clínicas.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from comun import (
    ROOT_DIR, TARGET, anos_walk_forward, cargar_matriz, columnas_predictoras, load_params,
    mascara_entrenamiento, resumir_predicciones,
)

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIG_PATH = REPORTS_DIR / "figures" / "xgboost_predicciones_cartagena.png"
PRED_PATH = REPORTS_DIR / "predicciones_xgboost.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def nuevo_modelo(params):
    return xgb.XGBClassifier(**params, objective="binary:logistic")


def entrenar_modelo_xgboost():
    print("=" * 70)
    print("XGBOOST - ALERTA TEMPRANA DE BROTE T+4 (WALK-FORWARD)")
    print("=" * 70)

    df = cargar_matriz()
    params = load_params("train")
    particion = load_params("particion")
    features = columnas_predictoras(df)
    print(f"Semanas: {len(df)} | Variables: {len(features)}")
    print(f"Hiperparámetros: {params}")

    horizonte = load_params("target")["horizonte"]
    predicciones = []
    for ano in anos_walk_forward(df, particion):
        train = df[mascara_entrenamiento(df["ANO"], ano, horizonte)]
        test = df[df["ANO"] == ano]
        modelo = nuevo_modelo(params).fit(train[features], train[TARGET])
        predicciones.append(pd.DataFrame({
            "ANO": test["ANO"], "SEMANA": test["SEMANA"], "y": test[TARGET].astype(int),
            "prob": modelo.predict_proba(test[features])[:, 1],
        }))
        print(f"  {ano}: entrenado con {len(train)} semanas (hasta {ano - 1})")

    pred = pd.concat(predicciones, ignore_index=True)
    pred.to_csv(PRED_PATH, index=False)

    # Modelo final para uso operativo: entrenado con toda la historia disponible
    modelo = nuevo_modelo(params).fit(df[features], df[TARGET])
    MODELS_DIR.mkdir(exist_ok=True)
    modelo.save_model(MODELS_DIR / "xgboost_model.json")

    resumen = resumir_predicciones(pred, particion)
    metrics = {k: resumen[k] for k in ("roc_auc_val", "roc_auc_test", "pr_auc_test", "roc_auc_por_ano")}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "-" * 50)
    print(f"Validación walk-forward (2020-2022) -> ROC-AUC: {metrics['roc_auc_val']:.4f}")
    print(f"Prueba walk-forward (2023-2025)     -> ROC-AUC: {metrics['roc_auc_test']:.4f}"
          f" | PR-AUC: {metrics['pr_auc_test']:.4f}")
    for ano, valor in metrics["roc_auc_por_ano"].items():
        print(f"    {ano}: {valor:.4f}")

    importancias = pd.Series(modelo.get_booster().get_score(importance_type="gain"))
    print("\nTop 10 variables (ganancia, modelo final):")
    print((importancias / importancias.sum()).sort_values(ascending=False).head(10).round(3).to_string())

    graficar(df, pred, particion)
    return modelo


def graficar(df, pred, particion):
    datos = pred[pred["ANO"] > particion["val_end_year"]].merge(
        df[["ANO", "SEMANA", "ratio_casos_umbral_objetivo"]], on=["ANO", "SEMANA"]
    )
    etiquetas = datos["ANO"].astype(str) + "-S" + datos["SEMANA"].astype(str)
    x = range(len(datos))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, 0, datos["y"], step="mid", color="#e07b39", alpha=0.25,
                    label="Brote real en t+4 (1 = sí)")
    ax.plot(x, datos["prob"], color="#2b6cb0", linewidth=1.2, label="Probabilidad de brote (XGBoost)")
    ax.set_xticks(list(x)[:: max(1, len(datos) // 12)])
    ax.set_xticklabels(etiquetas.iloc[:: max(1, len(datos) // 12)], rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidad")
    ax.set_xlabel("Semana en que se emite la alerta (año y semana epidemiológica)")
    ax.set_title("XGBoost walk-forward: alerta de brote a 4 semanas, Cartagena 2023-2025")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)
    print(f"\nGráfico guardado en: {FIG_PATH}")


if __name__ == "__main__":
    entrenar_modelo_xgboost()
