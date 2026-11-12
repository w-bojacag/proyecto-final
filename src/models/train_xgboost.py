"""
Entrenamiento de XGBoost para la alerta temprana de brote de dengue a T+4 en un municipio.

Protocolo walk-forward anual: para predecir el año Y se entrena con todos los años anteriores.
Las predicciones fuera de muestra de validación (2020-2022) y prueba (2023-2025) se guardan en
reports/municipios/<municipio>/predicciones_xgboost.csv, y el modelo de cada año en
models/<municipio>/walk_forward/ para poder reproducirlas.

Uso: python src/models/train_xgboost.py --municipio cartagena
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from comun import (
    TARGET, anos_walk_forward, cargar_matriz, carpeta_walk_forward, columnas_predictoras,
    load_params, mascara_entrenamiento, municipio_desde_argumentos, resumir_predicciones,
)


def nuevo_modelo(params):
    return xgb.XGBClassifier(**params, objective="binary:logistic")


def entrenar_modelo_xgboost(rutas):
    print("=" * 70)
    print(f"XGBOOST - ALERTA TEMPRANA DE BROTE T+4 (WALK-FORWARD) - {rutas.nombre}")
    print("=" * 70)

    df = cargar_matriz(rutas)
    params = load_params("train")
    particion = load_params("particion")
    horizonte = load_params("target")["horizonte"]
    features = columnas_predictoras(df)
    print(f"Semanas: {len(df)} | Variables: {len(features)}")

    carpeta_wf = carpeta_walk_forward(rutas, "xgboost")
    predicciones = []
    for ano in anos_walk_forward(df, particion):
        train = df[mascara_entrenamiento(df["ANO"], ano, horizonte)]
        test = df[df["ANO"] == ano]
        modelo = nuevo_modelo(params).fit(train[features], train[TARGET])
        modelo.save_model(carpeta_wf / f"{ano}.json")
        predicciones.append(pd.DataFrame({
            "ANO": test["ANO"], "SEMANA": test["SEMANA"], "y": test[TARGET].astype(int),
            "prob": modelo.predict_proba(test[features])[:, 1],
        }))
        print(f"  {ano}: entrenado con {len(train)} semanas (hasta {ano - 1})")

    pred = pd.concat(predicciones, ignore_index=True)
    pred.to_csv(rutas.reportes / "predicciones_xgboost.csv", index=False)

    # Modelo final para uso operativo: entrenado con toda la historia disponible
    modelo = nuevo_modelo(params).fit(df[features], df[TARGET])
    modelo.save_model(rutas.modelos / "xgboost.json")

    resumen = resumir_predicciones(pred, particion)
    metrics = {k: resumen[k] for k in ("roc_auc_val", "roc_auc_test", "pr_auc_test", "roc_auc_por_ano")}
    with open(rutas.reportes / "metrics_xgboost.json", "w", encoding="utf-8") as f:
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

    graficar(rutas, pred, particion)
    return modelo


def graficar(rutas, pred, particion):
    datos = pred[pred["ANO"] > particion["val_end_year"]].reset_index(drop=True)
    etiquetas = datos["ANO"].astype(str) + "-S" + datos["SEMANA"].astype(str)
    x = range(len(datos))
    paso = max(1, len(datos) // 12)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, 0, datos["y"], step="mid", color="#e07b39", alpha=0.25,
                    label="Brote real en t+4 (1 = sí)")
    ax.plot(x, datos["prob"], color="#2b6cb0", linewidth=1.2, label="Probabilidad de brote (XGBoost)")
    ax.set_xticks(list(x)[::paso])
    ax.set_xticklabels(etiquetas.iloc[::paso], rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidad")
    ax.set_xlabel("Semana en que se emite la alerta (año y semana epidemiológica)")
    ax.set_title(f"XGBoost walk-forward: alerta de brote a 4 semanas, {rutas.nombre.title()} 2023-2025")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    ruta = rutas.reportes / "xgboost_predicciones.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"\nGráfico guardado en: {ruta}")


if __name__ == "__main__":
    entrenar_modelo_xgboost(municipio_desde_argumentos("Entrena XGBoost para un municipio"))
