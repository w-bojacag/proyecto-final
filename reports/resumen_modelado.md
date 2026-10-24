# Resumen de Sesión: Estructuración, Feature Engineering y Modelado XGBoost

**Fecha:** Sábado, 24 de octubre de 2026  
**Proyecto:** Alertas Tempranas de Dengue (Bolívar y Meta)

---

## 1. Reorganización del Proyecto
Se implementó una estructura profesional estándar de Ciencia de Datos y Machine Learning:
- **`data/`**: Separación en `raw/` (Excel originales SIVIGILA), `processed/` (CSVs unificados y matrices de características).
- **`notebooks/`**: Cuadernos interactivos (`eda_bolivar.ipynb`).
- **`src/`**: Código fuente modular (`eda_bolivar.py`, `clima.py`, `cruce_ambiental_epidemiologico.py`, `features/build_features.py`, `models/train_xgboost.py`).
- **`reports/`**: Artefactos visuales (`figures/`) y documentación metodológica.

---

## 2. Ingeniería de Características (Feature Engineering)
* **Script:** `src/features/build_features.py`
* **Metodología:** Se generaron rezagos temporales (*lags* de 1 a 4 semanas) y promedios móviles de 4 semanas (`rolling_mean_4`) para las variables climáticas (precipitación, temperatura) y epidemiológicas.
* **Resultado:** Matriz de características limpia con **727 semanas** y **38 variables** guardada en `data/processed/matriz_caracteristicas_cartagena.csv`.

---

## 3. Modelado Predictivo con XGBoost
* **Script:** `src/models/train_xgboost.py`
* **Estrategia de Evaluación:** División cronológica estricta (80% entrenamiento, 20% prueba / Test) para evitar fuga de datos (*data leakage*).
* **Diagnóstico de Sobreajuste (Overfitting):**
  * *Modelo Base (sin regularización severa):* Train RMSE = 1.28 | Test RMSE = 25.30. Evidencia clara de memorización del pasado.
  * *Modelo Regularizado (`max_depth=3`, L1/L2):* Train RMSE = 7.47 | Test RMSE = 46.43. El error de prueba aumentó debido a un fuerte **cambio de distribución (*distribution shift*)** en los años recientes (2023-2025), donde ocurrieron picos epidémicos históricos sin precedentes que los modelos conservadores subestiman.
* **Variables más importantes:** Los rezagos de casos (`casos_confirmados_o_probables_lag_3` y medias móviles) dominan el poder predictivo debido a la inercia semanal de los brotes.

---

## 4. Próximos Pasos Sugeridos
1. Evaluar modelos de series temporales puras (como SARIMAX) o redes neuronales recurrentes (LSTM).
2. Definir umbrales de alerta epidemiológica (clasificación binaria: brote / no brote) en lugar de regresión pura de casos.
3. Sincronizar datos y código con DagsHub y GitHub.
