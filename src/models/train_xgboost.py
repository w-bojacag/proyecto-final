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

def crear_target_brote_t4(df, col_casos="casos_confirmados_o_probables", horizonte=4, anos_ventana=5):
    # Asegurar orden cronológico
    df = df.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
    
    # 1. Calcular el umbral dinámico (Canal Endémico Móvil)
    # Por cada semana en un año Y, el umbral es el Percentil 75 de esa misma semana
    # en los 'anos_ventana' años previos (ej. Y-5 a Y-1).
    umbrales = []
    for idx, row in df.iterrows():
        ano_actual = row["ANO"]
        semana_actual = row["SEMANA"]
        
        historia = df[(df["SEMANA"] == semana_actual) & 
                      (df["ANO"] < ano_actual) & 
                      (df["ANO"] >= ano_actual - anos_ventana)][col_casos]
        
        # Si hay al menos 3 años de historia, usamos el percentil móvil.
        # Si no (primeros años del dataset), caemos en el percentil global de esa semana.
        if len(historia) >= 3:
            umbral = historia.quantile(0.75)
        else:
            umbral = df[df["SEMANA"] == semana_actual][col_casos].quantile(0.75)
            
        umbrales.append(umbral)
        
    df["umbral_brote"] = umbrales
    
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
    casos_semana_actual = [col for col in df.columns if col.startswith("casos_") and "_lag_" not in col and "_roll_" not in col and "_tasa_" not in col]
    exclude_cols = ["ANO", "SEMANA", "umbral_brote", "es_brote_actual", "target_brote_t4"] + casos_semana_actual
    features = [col for col in df.columns if col not in exclude_cols]
    print(f"Variables excluidas (fuga del objetivo): {casos_semana_actual}")
    
    X = df[features]
    y = df[target]
    
    # ---------------------------------------------------------
    # VALIDACIÓN WALK-FORWARD (VENTANA DESLIZANTE ANUAL)
    # ---------------------------------------------------------
    train_end_year = params.get("train_end_year", 2019)
    val_end_year = params.get("val_end_year", 2022)
    test_years = sorted(df[df["ANO"] > val_end_year]["ANO"].unique())
    
    print(f"Iniciando Walk-Forward Validation para los años: {test_years}")
    
    all_y_test_real = []
    all_y_test_pred = []
    df_test_predictions = pd.DataFrame()
    
    # Calcular dinámicamente el peso para desbalance de clases (lo dejamos neutral como acordamos)
    scale_pos = 1.0  
    
    # Guardaremos el último modelo entrenado
    last_model = None
    
    for test_year in test_years:
        print(f"\n--- Entrenando para predecir el año {test_year} ---")
        # El conjunto de entrenamiento incluye TODO hasta el año anterior al de prueba actual
        mask_train = df["ANO"] < test_year
        mask_test_current = df["ANO"] == test_year
        
        X_train_wf, y_train_wf = X[mask_train], y[mask_train]
        X_test_wf, y_test_wf = X[mask_test_current], y[mask_test_current]
        
        print(f"Entrenamiento: {len(X_train_wf)} semanas (Hasta {test_year-1})")
        print(f"Prueba:        {len(X_test_wf)} semanas (Solo {test_year})")
        
        model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 80),
            learning_rate=params.get("learning_rate", 0.03),
            max_depth=params.get("max_depth", 2),
            min_child_weight=params.get("min_child_weight", 5),
            reg_alpha=params.get("reg_alpha", 2.0),
            reg_lambda=params.get("reg_lambda", 2.0),
            random_state=params.get("random_state", 42),
            subsample=params.get("subsample", 0.7),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            scale_pos_weight=scale_pos,
            objective="binary:logistic"
        )
        
        model.fit(X_train_wf, y_train_wf)
        last_model = model
        
        # Predicciones de este año
        y_pred_proba_wf = model.predict_proba(X_test_wf)[:, 1]
        
        # Guardar resultados
        all_y_test_real.extend(y_test_wf.values)
        all_y_test_pred.extend(y_pred_proba_wf)
        
        # Añadir al df de predicciones para el gráfico
        df_test_year = df[mask_test_current].copy()
        df_test_year["prediccion"] = y_pred_proba_wf
        df_test_predictions = pd.concat([df_test_predictions, df_test_year])
        
        auc_year = roc_auc_score(y_test_wf, y_pred_proba_wf)
        print(f"ROC-AUC para {test_year}: {auc_year:.4f}")
        
    # Guardar el último modelo entrenado (el que ya vio hasta 2024 para predecir 2025)
    model_path = MODELS_DIR / "xgboost_model.json"
    last_model.save_model(model_path)
    print(f"\nÚltimo modelo (Walk-Forward) guardado en: {model_path}")
    
    # ---------------------------------------------------------
    # MÉTRICAS GLOBALES DEL EXPERIMENTO WALK-FORWARD
    # ---------------------------------------------------------
    roc_test_global = float(roc_auc_score(all_y_test_real, all_y_test_pred))
    
    print("\n" + "-" * 50)
    print("DIAGNÓSTICO DE CLASIFICACIÓN (WALK-FORWARD T+4):")
    print("-" * 50)
    print(f"Prueba Global Acumulada (Test) -> ROC-AUC: {roc_test_global:.4f}")

    # Guardar métricas en JSON para DVC
    metrics = {
        "roc_auc_test_walk_forward": roc_test_global
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {METRICS_PATH}")
    
    # Importancia de características (del último modelo)
    importances = pd.Series(last_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n" + "-" * 40)
    print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES (Último Modelo):")
    print("-" * 40)
    print(importances.head(10))
    
    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df_test_predictions["ANO"].astype(str) + "-S" + df_test_predictions["SEMANA"].astype(str), df_test_predictions[target], label="Casos Reales", color="black", alpha=0.8)
    plt.plot(df_test_predictions["ANO"].astype(str) + "-S" + df_test_predictions["SEMANA"].astype(str), df_test_predictions["prediccion"], label="Predicción XGBoost (Walk-Forward)", color="red", linestyle="--", alpha=0.9)
    plt.xticks(rotation=45, ticks=range(0, len(df_test_predictions), max(1, len(df_test_predictions)//12)))
    plt.title("XGBoost Walk-Forward: Casos Reales vs Predichos de Dengue (Cartagena - Test Set)")
    plt.xlabel("Año y Semana Epidemiológica")
    plt.ylabel("Probabilidad / Brote")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    fig_path = FIG_DIR / "xgboost_predicciones_cartagena.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nGráfico guardado en: {fig_path}")
    plt.close()
    
    return last_model, roc_test_global


if __name__ == "__main__":
    modelo, roc_test_score = entrenar_modelo_xgboost()
