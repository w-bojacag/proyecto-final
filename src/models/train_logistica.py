"""
Regresión logística para la alerta temprana de brote de dengue a T+4 en un municipio.

Modelo interpretable: usa pocas variables, todas relativas al umbral de brote, y cada
coeficiente se puede leer como "cuánto cambia el riesgo de brote" (razón de odds).
Usa la misma etiqueta, las mismas semanas y el mismo walk-forward que XGBoost y LSTM.

Las variables se transforman con log1p (las razones casos/umbral son muy asimétricas) y se
estandarizan con las semanas de entrenamiento de cada iteración.

Uso: python src/models/train_logistica.py --municipio cartagena
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from comun import (
    TARGET, anos_walk_forward, cargar_matriz, carpeta_walk_forward, load_params,
    mascara_entrenamiento, municipio_desde_argumentos, resumir_predicciones,
)


def nuevo_modelo(params):
    return make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
        LogisticRegression(C=params["C"], max_iter=1000),
    )


def coeficientes(modelo, variables):
    escalador = modelo.named_steps["standardscaler"]
    reg = modelo.named_steps["logisticregression"]
    salida = {
        var: {
            "coeficiente": float(c),
            # Cuántas veces se multiplican las odds de brote al aumentar la variable en una
            # desviación estándar de log(1 + variable)
            "razon_odds_por_desviacion": float(np.exp(c)),
            "desviacion_log1p": float(s),
        }
        for var, c, s in zip(variables, reg.coef_[0], escalador.scale_)
    }
    salida["intercepto"] = float(reg.intercept_[0])
    return salida


def entrenar_logistica(rutas):
    print("=" * 70)
    print(f"REGRESIÓN LOGÍSTICA - ALERTA TEMPRANA DE BROTE T+4 (WALK-FORWARD) - {rutas.nombre}")
    print("=" * 70)

    df = cargar_matriz(rutas)
    params = load_params("logistica")
    particion = load_params("particion")
    horizonte = load_params("target")["horizonte"]
    variables = params["variables"]
    print(f"Variables: {variables} | C = {params['C']}")

    carpeta_wf = carpeta_walk_forward(rutas, "logistica")
    predicciones = []
    for ano in anos_walk_forward(df, particion):
        train = df[mascara_entrenamiento(df["ANO"], ano, horizonte)]
        test = df[df["ANO"] == ano]
        modelo = nuevo_modelo(params).fit(train[variables], train[TARGET])
        joblib.dump(modelo, carpeta_wf / f"{ano}.joblib")
        predicciones.append(pd.DataFrame({
            "ANO": test["ANO"], "SEMANA": test["SEMANA"], "y": test[TARGET].astype(int),
            "prob": modelo.predict_proba(test[variables])[:, 1],
        }))

    pred = pd.concat(predicciones, ignore_index=True)
    pred.to_csv(rutas.reportes / "predicciones_logistica.csv", index=False)

    # Modelo final con toda la historia: es el que se interpreta y el que se usaría en operación
    modelo = nuevo_modelo(params).fit(df[variables], df[TARGET])
    joblib.dump(modelo, rutas.modelos / "logistica.joblib")
    coefs = coeficientes(modelo, variables)
    with open(rutas.reportes / "logistica_coeficientes.json", "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=4)

    resumen = resumir_predicciones(pred, particion)
    metrics = {k: resumen[k] for k in ("roc_auc_val", "roc_auc_test", "pr_auc_test", "roc_auc_por_ano")}
    with open(rutas.reportes / "metrics_logistica.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nValidación walk-forward (2020-2022) -> ROC-AUC: {metrics['roc_auc_val']:.4f}")
    print(f"Prueba walk-forward (2023-2025)     -> ROC-AUC: {metrics['roc_auc_test']:.4f}"
          f" | PR-AUC: {metrics['pr_auc_test']:.4f}")
    for ano, valor in metrics["roc_auc_por_ano"].items():
        print(f"    {ano}: {valor:.4f}")
    print("\nCoeficientes del modelo final (razón de odds por desviación estándar):")
    for var in variables:
        print(f"  {var:35s} {coefs[var]['razon_odds_por_desviacion']:.2f}")


if __name__ == "__main__":
    entrenar_logistica(municipio_desde_argumentos("Entrena la regresión logística para un municipio"))
