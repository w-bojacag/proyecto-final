# Memoria Metodológica y Hallazgos del Proyecto (Para Informe Final)

Este documento es de uso estrictamente personal y no se subirá al repositorio de control de versiones. Su propósito es servir de insumo o borrador para redactar los hallazgos en el informe final de tesis (ej. en `informe_avance.docx`).

## 1. Arquitectura del Pipeline de Machine Learning
El proyecto utiliza **DVC (Data Version Control)** para garantizar la reproducibilidad de los experimentos epidemiológicos. Las etapas consolidadas son:
1. **`clima`**: Descarga de datos ambientales (GEE) para el polígono urbano.
2. **`cruce`**: Fusión de datos meteorológicos semanales con reportes de dengue (SIVIGILA).
3. **`features`**: Ingeniería de variables temporales (Rezagos temporales - *Lags*, Medias móviles espaciales y temporales).
4. **`train`**: Entrenamiento del modelo predictivo (XGBoost) manejando hiperparámetros estáticos desde `params.yaml`.
5. **`evaluate`**: Evaluación de clasificación clínica (Matriz de confusión) sobre el conjunto de prueba ciego (2023-2025).

## 2. Ajuste de Hiperparámetros (Evitar Sobreajuste)
* Inicialmente, variaciones en `build_features.py` indujeron a un fuerte *overfitting* (AUC > 0.96 en entrenamiento, pero cayendo drásticamente en validación/prueba).
* Se utilizaron técnicas de regularización en el XGBoost (`max_depth: 1`, regularización L1/L2 a través de `reg_alpha` y `reg_lambda`, e iteraciones controladas).

## 3. Matriz de Confusión y el Índice de Youden (Hallazgo Principal)
El modelo de clasificación binaria ("Brote" vs "Normal") requería un **umbral de decisión clínico**.

1. **Trade-off Sensibilidad vs. Falsos Positivos:**
   * Un umbral estático en **0.5**: 95.8% Sensibilidad / 31 Falsos Positivos. (Modelo "Paranoico", alto costo al sistema de salud).
   * Un umbral estricto en **0.7**: 0% Sensibilidad / 0 FP. (Modelo conservador e inútil, no detectó nada).

2. **Implementación del Índice de Youden ($J$):**
   * Metodológicamente, se implementó un script que **aprende el umbral ideal** exclusivamente basándose en la Curva ROC del conjunto de **Validación (2020-2022)**, evitando el sesgo o fuga de información.
   * La fórmula utilizada fue: $J = Sensibilidad (TPR) - Tasa de Falsos Positivos (FPR)$.
   * El umbral óptimo calculado fue **0.4455**.

3. **La Fase Híper-Endémica de Cartagena (2023-2025):**
   * Al aplicar ciegamente el umbral óptimo de 0.4455 al conjunto de **Prueba (2023-2025)**, el modelo clasificó 149 de 153 semanas (casi el 100%) como alertas de Brote (TN = 0).
   * **Conclusión Epidemiológica:** Los niveles base (normales) de casos de dengue en Cartagena experimentaron un alza estructural sin precedentes después de la pandemia de COVID-19. 
   * **Implicación Metodológica:** Al calcular el objetivo ("Umbral Percentil 75") tomando en cuenta **toda la historia histórica (2007-2019)**, los datos pos-pandémicos superan ese umbral constantemente.

## 4. Próximos Pasos (Hoja de Ruta)
1. **Canal Endémico Móvil/Dinámico:** En lugar de calcular el Percentil 75 sobre los 15 años de historia completa, se propondrá modificar el cálculo de la variable objetivo (`target_brote_t4`) usando una ventana móvil (ej. promedios semanales de los últimos 5 a 7 años), adaptando la sensibilidad del modelo al comportamiento endémico reciente.
2. **Despliegue a Bolívar:** Extender la estructura del proyecto DVC para modelar los municipios más afectados de Bolívar y observar el comportamiento geográfico de los brotes.

## 5. Implementación del Canal Endémico Móvil
* **Decisión de Diseño:** En `train_xgboost.py`, se modificó la función `crear_target_brote_t4`. El cálculo del "Percentil 75 global" fue reemplazado por un **Percentil 75 móvil de 5 años**. Es decir, para evaluar si la semana 10 de 2024 es un brote, el modelo busca el percentil 75 de las semanas 10 comprendidas entre 2019 y 2023 exclusivamente.
* **Impacto Clínico:** 
  * Los Casos Reales de Brote (Test) bajaron matemáticamente de 119 a **79**, pues el umbral exigido subió al nivel de la hiperendemia post-COVID.
  * El modelo logró una **Sensibilidad del 92.4%** (detectó 73 de 79 brotes a tiempo).
  * Los **Falsos Positivos subieron a 72**, lo cual revela que el XGBoost está perdiendo poder de discriminación (AUC Test cayó a 0.49). Esto sugiere que predecir a T+4 en escenarios altamente inestables requiere incluir más rezagos históricos de casos o repensar la importancia del clima.

## 6. Inyección de Variables de Velocidad y Estacionalidad
* **Diagnóstico:** El modelo estaba perdiendo poder predictivo bajo el Canal Endémico Móvil porque XGBoost descartaba inadvertidamente las variables de **tasa de cambio** (velocidad epidémica) en `train_xgboost.py` debido a una regla de exclusión de fugas demasiado estricta. Además, carecía de noción de estacionalidad (meses/trimestres).
* **Solución Técnica:**
  1. Se corrigió `train_xgboost.py` (y `evaluate_model.py`) para que incluyan las variables terminadas en `_tasa_cambio_1` (velocidad a 1 semana) y `_tasa_cambio_4` (velocidad a 1 mes).
  2. En `build_features.py` se añadieron las columnas de `mes_aproximado` y `trimestre_aproximado`.
* **Resultados Clínicos (Mejora del ROC-AUC):**
  * Las tasas de cambio irrumpieron inmediatamente en el **Top 10 de características más importantes** (posiciones 4 y 5), demostrando que XGBoost necesitaba "ver" la aceleración de los casos, no solo el volumen absoluto.
  * El **ROC-AUC en validación subió de 0.76 a 0.79**, y en **Test recuperó terreno de 0.49 a 0.57**.
  * A nivel de matriz de confusión (Youden: 0.4267), los Falsos Positivos se mantuvieron en 72, indicando que el modelo aún prioriza no dejar pasar ningún brote, pero su capacidad de ranqueo (AUC) va mejorando.

## 7. Límite de la Regularización y Desactivación del scale_pos_weight
* Para intentar reducir las alarmas falsas, se aplicó una regularización extrema (L1/L2 a 20.0, learning rate 0.005, min_child_weight 15).
* **Diagnóstico Metodológico:** La agresiva regularización estaba siendo contrarrestada por el parámetro `scale_pos_weight` calculado dinámicamente. Este multiplicador forzaba matemáticamente al modelo a temerle tanto a un Falso Negativo que prefería lanzar alertas indiscriminadas.
* **Solución y Hallazgo Final:** Se desactivó el sesgo artificial (`scale_pos = 1.0`). Como resultado, las probabilidades predichas se normalizaron y el **AUC en Validación alcanzó su máximo histórico (0.815)**.
* **Problema Persistente:** En el conjunto de Prueba (2023-2025), el AUC volvió a caer (0.53) y los Falsos Positivos se mantuvieron en 74 (True Negatives = 0). Al ver la importancia de variables, el `roll_mean_4` acapara el 99% del peso. 
* **Conclusión Definitiva sobre Cartagena:** El clima ha dejado de ser predictivo a T+4 en la actual fase de hiperendemia. El modelo XGBoost se convirtió en un "predictor ingenuo" (Naïve forecaster) que simplemente repite los casos del mes pasado, asumiendo que si la hiperendemia sigue, siempre habrá brote. Predecir a 4 semanas requerirá una nueva métrica epidemiológica que no se sature.
