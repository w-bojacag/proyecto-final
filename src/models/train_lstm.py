"""
Script de entrenamiento y evaluación de un modelo LSTM para Alerta Temprana de Dengue.
Utiliza TensorFlow/Keras para modelar la serie temporal como secuencias.
"""

import os
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evitar advertencias innecesarias de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data_generator import crear_ventanas_temporales

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
# Importante: Usamos la matriz base, el LSTM aprenderá sus propios rezagos temporales
INPUT_PATH = ROOT_DIR / "data" / "processed" / "matriz_cruce_semanal_cartagena.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = ROOT_DIR / "reports" / "metrics_lstm.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_params():
    params_path = ROOT_DIR / "params.yaml"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            # Podríamos crear una sección 'lstm' en params, por ahora cargamos default o train
            return yaml.safe_load(f).get("lstm", {
                "window_size": 12,
                "epochs": 100,
                "batch_size": 32,
                "lstm_units": 16,      # Arquitectura más sencilla (16 en lugar de 32)
                "dropout_rate": 0.5,   # Mayor agresividad apagando neuronas para evitar memorización
                "learning_rate": 0.001,
                "anos_ventana": 5,
                "train_end_year": 2019,
                "val_end_year": 2022
            })
    return {}

def crear_target_binario(df, col_casos="casos_confirmados_o_probables", anos_ventana=5):
    """
    Calcula el Canal Endémico Móvil para clasificar cada semana como Brote (1) o Normal (0).
    A diferencia de XGBoost, no hacemos shift aquí; el data_generator se encarga del horizonte temporal.
    """
    df = df.sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
    umbrales = []
    
    for idx, row in df.iterrows():
        ano_actual = row["ANO"]
        semana_actual = row["SEMANA"]
        
        historia = df[(df["SEMANA"] == semana_actual) & 
                      (df["ANO"] < ano_actual) & 
                      (df["ANO"] >= ano_actual - anos_ventana)][col_casos]
        
        if len(historia) >= 3:
            umbral = historia.quantile(0.75)
        else:
            umbral = df[df["SEMANA"] == semana_actual][col_casos].quantile(0.75)
            
        umbrales.append(umbral)
        
    df["umbral_brote"] = umbrales
    df["es_brote_actual"] = (df[col_casos] > df["umbral_brote"]).astype(int)
    return df

def entrenar_modelo_lstm():
    print("=" * 70)
    print("ENTRENAMIENTO DE RED NEURONAL LSTM - ALERTA TEMPRANA DE DENGUE")
    print("=" * 70)
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo base en: {INPUT_PATH}")
        
    df = pd.read_csv(INPUT_PATH)
    params = load_params()
    print(f"Hiperparámetros LSTM: {params}")
    
    # 1. Crear Target actual
    df = crear_target_binario(df, anos_ventana=params.get("anos_ventana", 5))
    
    # 2. Seleccionar Features a pasar a la red
    # La red neuronal se beneficia de variables puras (ella abstrae las relaciones temporales).
    # Excluimos identificadores y la variable objetivo actual/futura.
    features = [
        "precipitation_mm", "lst_day_c", "lst_night_c", "air_temp_max_c", 
        "air_temp_mean_c", "air_temp_min_c", "dewpoint_c", "evi", "ndvi",
        "casos_confirmados_o_probables" # Le pasamos los casos actuales para que la LSTM entienda la curva epidémica
    ]
    
    # Asegurar que existan
    features = [f for f in features if f in df.columns]
    
    df[features] = df[features].interpolate(method='linear', limit_direction='both').fillna(0)
    
    # Normalización Híbrida:
    # 1. Escalado global para variables climáticas
    climatic_features = [f for f in features if f != "casos_confirmados_o_probables"]
    scaler = StandardScaler()
    df[climatic_features] = scaler.fit_transform(df[climatic_features])
    
    # 2. Transformación logarítmica (np.log1p) para los casos epidemiológicos
    # Esto comprime la escala hiperendémica sin perder la magnitud absoluta
    if "casos_confirmados_o_probables" in features:
        df["casos_confirmados_o_probables"] = np.log1p(df["casos_confirmados_o_probables"])
    
    # 3. Generar Ventanas Temporales (Tensores)
    window_size = params.get("window_size", 12)
    horizonte = 4
    
    X, y, df_fechas = crear_ventanas_temporales(df, features, "es_brote_actual", window_size, horizonte)
    
    print(f"Forma del Tensor X (Features): {X.shape} -> (Muestras, Pasos de tiempo, Variables)")
    print(f"Forma del Vector y (Target): {y.shape}")
    
    # 4. Partición Temporal
    train_end_year = params.get("train_end_year", 2019)
    val_end_year = params.get("val_end_year", 2022)
    
    mask_train = df_fechas["ANO"] <= train_end_year
    mask_val = (df_fechas["ANO"] > train_end_year) & (df_fechas["ANO"] <= val_end_year)
    mask_test = df_fechas["ANO"] > val_end_year
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]
    
    # 5. Construcción del Modelo LSTM
    model = Sequential([
        LSTM(params.get("lstm_units", 32), input_shape=(window_size, len(features))),
        Dropout(params.get("dropout_rate", 0.4)),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    
    print("\nResumen de la Arquitectura LSTM:")
    model.summary()
    
    # Early Stopping para evitar sobreajuste
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # 6. Entrenamiento
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        epochs=params.get("epochs", 100),
        batch_size=params.get("batch_size", 32),
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Guardar modelo
    model_path = MODELS_DIR / "lstm_model.keras"
    model.save(model_path)
    print(f"\nModelo LSTM guardado en: {model_path}")
    
    # 7. Predicciones y Evaluación ROC-AUC
    y_pred_train = model.predict(X_train).ravel()
    y_pred_val = model.predict(X_val).ravel()
    y_pred_test = model.predict(X_test).ravel()
    
    roc_train = float(roc_auc_score(y_train, y_pred_train))
    roc_val = float(roc_auc_score(y_val, y_pred_val))
    roc_test = float(roc_auc_score(y_test, y_pred_test))
    
    print("\n" + "-" * 50)
    print("DIAGNÓSTICO DE CLASIFICACIÓN LSTM (ALERTA T+4):")
    print("-" * 50)
    print(f"Entrenamiento -> ROC-AUC: {roc_train:.4f}")
    print(f"Validación    -> ROC-AUC: {roc_val:.4f}")
    print(f"Prueba (Test) -> ROC-AUC: {roc_test:.4f}")
    
    metrics = {
        "roc_auc_train": roc_train,
        "roc_auc_val": roc_val,
        "roc_auc_test": roc_test
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    entrenar_modelo_lstm()
