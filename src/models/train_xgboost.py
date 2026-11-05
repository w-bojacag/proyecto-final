"""
Script de entrenamiento y evaluación de un modelo XGBoost para Alerta Temprana de Dengue (Cartagena).
Integra carga de hiperparámetros desde params.yaml y guardado de métricas para DVC.
"""

from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_caracteristicas_cartagena.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_params():
    params_path = ROOT_DIR / "params.yaml"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("train", {})
    return {}

def crear_target_brote_t4(df, col_casos="casos_confirmados_o_probables", horizonte=4):
    # Asegurar orden cronológico
    df = df.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
    
    # 1. Calcular el Percentil 75 histórico por cada semana epidemiológica (1-52)
    # Usamos los datos históricos para calcular el umbral normal de cada época del año
    umbrales_semanales = df.groupby("SEMANA")[col_casos].quantile(0.75).to_dict()
    df["umbral_brote"] = df["SEMANA"].map(umbrales_semanales)
    
    # 2. Definir si la semana real supera el umbral (Brote = 1, Normal = 0)
    df["es_brote_actual"] = (df[col_casos] > df["umbral_brote"]).astype(int)
    
    # 3. Desplazar el objetivo 4 semanas hacia el FUTURO (t + 4)
    # Esto significa: en la fila actual (semana t), ¿habrá brote en t + 4?
    df["target_brote_t4"] = df["es_brote_actual"].shift(-horizonte)
    
    # Eliminar las últimas 4 filas que quedarán con NaN por el shift hacia adelante
    df = df.dropna(subset=["target_brote_t4"]).reset_index(drop=True)
    
    return df

def entrenar_modelo_xgboost():
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO XGBOOST - ALERTA TEMPRANA DE DENGUE")
    print("=" * 70)
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró la matriz de características en: {INPUT_PATH}")
        
    df = pd.read_csv(INPUT_PATH)
    print(f"Total de registros cargados: {len(df)}")
    
    # Cargar hiperparámetros desde params.yaml
    params = load_params()
    print(f"Hiperparámetros cargados: {params}")
   
    # Cálculo del Target de Brote Y_t+4
    df = crear_target_brote_t4(df)
    print(f"se aplica cambio de Target en df")
    
    target = "target_brote_t4"
    # Los conteos de casos de la misma semana (casos_totales, casos_confirmados, ...) contienen
    # el objetivo y no se conocen al momento de predecir: solo se usan sus rezagos y medias móviles.
    casos_semana_actual = [col for col in df.columns if col.startswith("casos_") and "_lag_" not in col and "_roll_" not in col]
    exclude_cols = ["ANO", "SEMANA", "umbral_brote", "es_brote_actual", "target_brote_t4"] + casos_semana_actual
    features = [col for col in df.columns if col not in exclude_cols]
    print(f"Variables excluidas (fuga del objetivo): {casos_semana_actual}")
    
    X = df[features]
    y = df[target]
    
    # Partición cronológica por años: los hiperparámetros se ajustan con validación;
    # prueba se reserva para la evaluación final.
    train_end_year = params.get("train_end_year", 2019)
    val_end_year = params.get("val_end_year", 2022)
    mask_train = df["ANO"] <= train_end_year
    mask_val = (df["ANO"] > train_end_year) & (df["ANO"] <= val_end_year)
    mask_test = df["ANO"] > val_end_year

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    df_test = df[mask_test].copy()

    def rango(mask):
        d = df[mask]
        return f"{int(d['ANO'].iloc[0])}-S{int(d['SEMANA'].iloc[0])} a {int(d['ANO'].iloc[-1])}-S{int(d['SEMANA'].iloc[-1])}"

    print(f"Entrenamiento: {len(X_train)} semanas ({rango(mask_train)})")
    print(f"Validación:    {len(X_val)} semanas ({rango(mask_val)})")
    print(f"Prueba (Test): {len(X_test)} semanas ({rango(mask_test)})")
    
    # Inicializar y entrenar XGBoost Regressor
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 80),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 3),
        min_child_weight=params.get("min_child_weight", 3),
        reg_alpha=params.get("reg_alpha", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
        subsample=params.get("subsample", 0.7),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        objective="binary:logistic"
    )
    
    model.fit(X_train, y_train)
    
    # Guardar modelo entrenado
    model_path = MODELS_DIR / "xgboost_model.json"
    model.save_model(model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Predicciones
    #y_pred_train = model.predict(X_train)
    #y_pred_val = model.predict(X_val)
    #y_pred_test = model.predict(X_test)

    # Predicciones de probabilidad
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    # Métricas
    #rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    #mae_train = float(mean_absolute_error(y_train, y_pred_train))

    #rmse_val = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    #mae_val = float(mean_absolute_error(y_val, y_pred_val))

    #rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    #mae_test = float(mean_absolute_error(y_test, y_pred_test))

    # Métrica de rendimiento principal para alertas tempranas
    roc_train = float(roc_auc_score(y_train, y_pred_train))
    roc_val = float(roc_auc_score(y_val, y_pred_val))
    roc_test = float(roc_auc_score(y_test, y_pred_test))

    print("\n" + "-" * 50)
    print("DIAGNÓSTICO DE CLASIFICACIÓN (ALERTA DE BROTE T+4):")
    print("-" * 50)
    print(f"Entrenamiento -> ROC-AUC: {roc_train:.4f}")
    print(f"Validación    -> ROC-AUC: {roc_val:.4f}")
    print(f"Prueba (Test) -> ROC-AUC: {roc_test:.4f}")

    # Guardar métricas en JSON para DVC
    metrics = {
        "roc_auc_train": roc_train,
        "roc_auc_val": roc_val,
        "roc_auc_test": roc_test
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {METRICS_PATH}")
    
    # Importancia de características
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n" + "-" * 40)
    print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print("-" * 40)
    print(importances.head(10))
    
    # Gráfico
    df_test["prediccion"] = y_pred_test
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_test["ANO"].astype(str) + "-S" + df_test["SEMANA"].astype(str), df_test[target], label="Casos Reales", color="black", alpha=0.8)
    plt.plot(df_test["ANO"].astype(str) + "-S" + df_test["SEMANA"].astype(str), df_test["prediccion"], label="Predicción XGBoost", color="red", linestyle="--", alpha=0.9)
    plt.xticks(rotation=45, ticks=range(0, len(df_test), max(1, len(df_test)//12)))
    plt.title("XGBoost: Casos Reales vs Predichos de Dengue (Cartagena - Test Set)")
    plt.xlabel("Año y Semana Epidemiológica")
    plt.ylabel("Número de casos")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    fig_path = FIG_DIR / "xgboost_predicciones_cartagena.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nGráfico guardado en: {fig_path}")
    plt.close()
    
    return model, roc_test


if __name__ == "__main__":
    modelo, roc_test_score = entrenar_modelo_xgboost()
