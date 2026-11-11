"""
Script de evaluación clínica de las predicciones de XGBoost.
Calcula métricas de decisión binaria: Sensibilidad (Recall), Precisión, Falsos Positivos.
El umbral se determina dinámicamente maximizando el Índice de Youden en el conjunto de Validación.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_curve
from train_xgboost import crear_target_brote_t4, load_params

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_caracteristicas_cartagena.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_model.json"
METRICS_PATH = ROOT_DIR / "reports" / "eval_metrics.json"

def evaluar_modelo():
    print("=" * 70)
    print("EVALUACIÓN CLÍNICA DEL MODELO - CONJUNTO DE PRUEBA")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {MODEL_PATH}")
        
    df = pd.read_csv(INPUT_PATH)
    params = load_params()
    train_end_year = params.get("train_end_year", 2019)
    val_end_year = params.get("val_end_year", 2022)
    
    df = crear_target_brote_t4(df)
    
    casos_semana_actual = [col for col in df.columns if col.startswith("casos_") and "_lag_" not in col and "_roll_" not in col and "_tasa_" not in col]
    exclude_cols = ["ANO", "SEMANA", "umbral_brote", "es_brote_actual", "target_brote_t4"] + casos_semana_actual
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Conjunto de Validación (para hallar el umbral óptimo)
    mask_val = (df["ANO"] > train_end_year) & (df["ANO"] <= val_end_year)
    X_val = df[mask_val][features]
    y_val = df[mask_val]["target_brote_t4"]

    # Conjunto de Prueba
    mask_test = df["ANO"] > val_end_year
    X_test = df[mask_test][features]
    y_test = df[mask_test]["target_brote_t4"]
    
    # Cargar el modelo guardado
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # 1. ENCONTRAR UMBRAL ÓPTIMO EN VALIDACIÓN (Índice de Youden)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_val)
    
    # Índice de Youden = Sensibilidad (TPR) + Especificidad (1 - FPR) - 1
    # Equivale a maximizar TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Umbral óptimo calculado en Validación (Youden): {optimal_threshold:.4f}")
    
    # 2. APLICAR UMBRAL EN PRUEBA (TEST)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_pred_proba_test >= optimal_threshold).astype(int)
    
    # Cálculo de métricas
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    sensibilidad = recall_score(y_test, y_pred_bin, zero_division=0)
    precision = precision_score(y_test, y_pred_bin, zero_division=0)
    
    print("-" * 30)
    print(f"Casos Reales de Brote (Test): {sum(y_test)}")
    print(f"Alertas Emitidas (Test): {sum(y_pred_bin)}")
    print("-" * 30)
    print(f"Sensibilidad (Recall): {sensibilidad:.2%} -> (Brotes detectados a tiempo)")
    print(f"Precisión: {precision:.2%} -> (Alertas que fueron reales)")
    print(f"Falsos Positivos (FP): {fp} -> (Semanas con alertas falsas)")
    print(f"Verdaderos Positivos (TP): {tp}")
    print(f"Falsos Negativos (FN): {fn}")
    print(f"Verdaderos Negativos (TN): {tn}")
    print("=" * 70)
    
    metrics = {
        "umbral_optimo": float(optimal_threshold),
        "sensibilidad": float(sensibilidad),
        "precision": float(precision),
        "falsos_positivos": int(fp),
        "verdaderos_positivos": int(tp),
        "falsos_negativos": int(fn),
        "verdaderos_negativos": int(tn)
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Métricas de evaluación guardadas en: {METRICS_PATH}")

if __name__ == "__main__":
    evaluar_modelo()
