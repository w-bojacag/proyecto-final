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
    
    target = "casos_confirmados_o_probables"
    exclude_cols = ["ANO", "SEMANA", target]
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df[target]
    
    test_size = params.get("test_size", 0.20)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Tamaño conjunto de entrenamiento: {len(X_train)} semanas")
    print(f"Tamaño conjunto de prueba (Test): {len(X_test)} semanas")
    
    # Inicializar y entrenar XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 80),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 3),
        min_child_weight=params.get("min_child_weight", 3),
        reg_alpha=params.get("reg_alpha", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
        subsample=params.get("subsample", 0.7),
        colsample_bytree=params.get("colsample_bytree", 0.7)
    )
    
    model.fit(X_train, y_train)
    
    # Guardar modelo entrenado
    model_path = MODELS_DIR / "xgboost_model.json"
    model.save_model(model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    mae_train = float(mean_absolute_error(y_train, y_pred_train))
    
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    
    print("\n" + "-" * 50)
    print("DIAGNÓSTICO DE SOBREAJUSTE: COMPARACIÓN TRAIN VS TEST")
    print("-" * 50)
    print(f"Entrenamiento -> RMSE: {rmse_train:.2f}  |  MAE: {mae_train:.2f}")
    print(f"Prueba (Test)   -> RMSE: {rmse_test:.2f}  |  MAE: {mae_test:.2f}")
    
    # Guardar métricas en JSON para DVC
    metrics = {
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "rmse_test": rmse_test,
        "mae_test": mae_test
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
    
    return model, rmse_test, mae_test


if __name__ == "__main__":
    entrenar_modelo_xgboost()
