"""
Script de entrenamiento y evaluación de un modelo XGBoost para Alerta Temprana de Dengue (Cartagena).
Objetivo: Predecir los casos de dengue de la semana actual utilizando rezagos climáticos y epidemiológicos.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_caracteristicas_cartagena.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def entrenar_modelo_xgboost():
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO XGBOOST - ALERTA TEMPRANA DE DENGUE")
    print("=" * 70)
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró la matriz de características en: {INPUT_PATH}")
        
    df = pd.read_csv(INPUT_PATH)
    print(f"Total de registros cargados: {len(df)}")
    
    # Definir la variable objetivo (Target) y las características (Features)
    # Target: casos_confirmados_o_probables en la semana t
    target = "casos_confirmados_o_probables"
    
    # Excluir columnas no predictivas o fechas directas si interfieren
    exclude_cols = ["ANO", "SEMANA", target]
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df[target]
    
    # División cronológica (Train / Test)
    # Por ejemplo, usar el 80% más antiguo para entrenamiento y el 20% más reciente para prueba
    split_idx = int(len(df) * 0.80)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Tamaño conjunto de entrenamiento: {len(X_train)} semanas")
    print(f"Tamaño conjunto de prueba (Test): {len(X_test)} semanas")
    
    # Inicializar y entrenar XGBoost Regressor con regularización
    model = xgb.XGBRegressor(
        n_estimators=80,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        subsample=0.7,
        colsample_bytree=0.7
    )
    
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluar métricas en Train y Test
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print("\n" + "-" * 50)
    print("DIAGNÓSTICO DE SOBREAJUSTE: COMPARACIÓN TRAIN VS TEST")
    print("-" * 50)
    print(f"Entrenamiento -> RMSE: {rmse_train:.2f}  |  MAE: {mae_train:.2f}")
    print(f"Prueba (Test)   -> RMSE: {rmse_test:.2f}  |  MAE: {mae_test:.2f}")
    
    if rmse_train < (rmse_test * 0.6):
        print("\n[ALERTA] Posible sobreajuste detectado: El error en entrenamiento es significativamente menor que en prueba.")
    else:
        print("\n[OK] Generalización adecuada: Los errores entre entrenamiento y prueba son comparables, sin evidencia severa de sobreajuste.")
    
    # Importancia de características (Feature Importance)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n" + "-" * 40)
    print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print("-" * 40)
    print(importances.head(10))
    
    # Graficar resultados (Real vs Predicho en Test)
    df_test["prediccion"] = y_pred_test
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_test["ANO"].astype(str) + "-S" + df_test["SEMANA"].astype(str), df_test[target], label="Casos Reales", color="black", alpha=0.8)
    plt.plot(df_test["ANO"].astype(str) + "-S" + df_test["SEMANA"].astype(str), df_test["prediccion"], label="Predicción XGBoost", color="red", linestyle="--", alpha=0.9)
    plt.xticks(rotation=45, ticks=range(0, len(df_test), max(1, len(df_test)//12)))
    plt.title("XGBoost: Casos Reales vs Predichos de Dengue (Cartagena - Test Set)")
    plt.xlabel("Año y Semana Epidemiológica")
    plt.ylabel("Número de Casos")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    fig_path = FIG_DIR / "xgboost_predicciones_cartagena.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nGráfico guardado exitosamente en: {fig_path}")
    plt.close()
    
    return model, rmse_test, mae_test


if __name__ == "__main__":
    entrenar_modelo_xgboost()
