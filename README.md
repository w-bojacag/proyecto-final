# Sistema de alerta temprana de dengue en Bolívar

Predice con 4 semanas de anticipación si habrá brote de dengue en cinco municipios de Bolívar
(Colombia), a partir de los casos notificados a SIVIGILA y del clima de cada cabecera urbana.

- **Municipios:** Cartagena, Magangué, El Carmen de Bolívar, Turbaco y Santa Rosa del Sur, con un
  modelo independiente por municipio.
- **Brote:** semana en que los casos superan el canal endémico de Bortman (límite superior del IC 95 %
  de la media geométrica de los 5 años anteriores).
- **Modelo operativo:** regresión logística con 3 variables. XGBoost y LSTM se mantienen como
  modelos de comparación.
- **Reproducible:** todo el pipeline corre con [DVC](https://dvc.org). Los datos y modelos están en
  [DagsHub](https://dagshub.com/w-bojacag/proyecto-final).

## Resultados vigentes

Regresión logística, prueba 2023–2025. Cada año se predice con un modelo entrenado solo con los
años anteriores (walk-forward).

| Municipio | AUC prueba [IC 95 %] | Brotes avisados | Anticipación mediana (semanas) |
|---|---|---|---|
| Cartagena | 0,984 [0,95–1,00] | 4 de 7 | 7 |
| Magangué | 0,959 [0,92–0,99] | 12 de 13 | 6 |
| El Carmen de Bolívar | 0,962 [0,90–1,00] | 2 de 2 | 6 |
| Santa Rosa del Sur | 0,948 [0,88–1,00] | 6 de 8 | 24 |
| Turbaco | 0,842 [0,75–0,93] | 5 de 12 | 4 |

Un brote cuenta como avisado si hubo una alerta entre 2 semanas antes de su inicio y su inicio.
Las tablas completas, con XGBoost, LSTM y líneas base, están en
[reports/resumen_municipios.csv](reports/resumen_municipios.csv).

## Instalación

Requisitos: Python 3.12 y Git. Los comandos son para Windows; en Linux o macOS, cambia
`.venv\Scripts\` por `.venv/bin/`.

```bash
git clone https://github.com/w-bojacag/proyecto-final.git
cd proyecto-final
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las etapas de `dvc.yaml` llaman a `.venv\Scripts\python.exe`, así que el entorno virtual debe
llamarse `.venv` y estar en la raíz del proyecto.

### Acceso a los datos (DagsHub)

Los datos y modelos no están en Git; están en el remoto de DVC en DagsHub. El repositorio es
público, así que para **descargarlos** no se necesita cuenta:

```bash
.venv\Scripts\dvc.exe pull
```

Para **subir** resultados (`dvc push`) hace falta ser colaborador:

1. Crea una cuenta en [DagsHub](https://dagshub.com) y pide que te agreguen como colaborador del
   repositorio.
2. Genera tu propio token en DagsHub (*User settings → Tokens*). **Cada persona usa su token;
   no se comparten.**
3. Configúralo solo en tu copia local (queda en `.dvc/config.local`, que Git no sube):

   ```bash
   .venv\Scripts\dvc.exe remote modify origin --local auth basic
   .venv\Scripts\dvc.exe remote modify origin --local user <tu-usuario-dagshub>
   .venv\Scripts\dvc.exe remote modify origin --local password <tu-token>
   ```

## Uso

```bash
.venv\Scripts\dvc.exe repro                       # corre lo que haya cambiado
.venv\Scripts\dvc.exe repro evaluate@turbaco      # un municipio hasta su evaluación
.venv\Scripts\dvc.exe metrics show                # métricas de todos los municipios
```

- Después de `dvc pull`, `dvc repro` no debería tener nada que correr. Si cambias código o
  parámetros, solo se vuelven a correr las etapas afectadas.
- Reentrenar los cinco municipios toma unos 30 minutos; la mayor parte es el LSTM.
- Para subir resultados nuevos: `git commit` y `git push` para el código y los `.lock`;
  `dvc push` para datos y modelos.

## Pipeline

Las etapas con `foreach` se repiten por municipio y aparecen como `<etapa>@<municipio>`
(por ejemplo `train_logistica@magangue`).

| Etapa | Script | Qué hace |
|---|---|---|
| `eda` | `src/eda_bolivar.py` | Calidad de datos y series del top 5 de municipios de Bolívar. |
| `clima` | `src/clima.py` | Clima diario 2007–2025 de la cabecera urbana desde Google Earth Engine. **Congelada:** usa el CSV guardado en DagsHub. |
| `cruce` | `src/cruce_ambiental_epidemiologico.py` | Une casos y clima por semana epidemiológica del INS. |
| `features` | `src/features/build_features.py` | Rezagos, medias móviles, canal endémico, etiqueta de brote a 4 semanas y variables relativas al umbral. |
| `train_logistica` | `src/models/train_logistica.py` | Regresión logística (modelo operativo) con walk-forward anual. |
| `train` | `src/models/train_xgboost.py` | XGBoost con walk-forward anual. |
| `train_lstm` | `src/models/train_lstm.py` | LSTM con walk-forward anual. |
| `evaluate` | `src/models/evaluate_model.py` | Compara modelos y líneas base: AUC con IC por bootstrap de bloques y métricas por brote. |
| `resumen` | `src/models/resumen_municipios.py` | Tabla comparativa de todos los municipios. |

Volver a extraer el clima requiere una cuenta de Google Earth Engine:
`dvc repro --force clima@<municipio>`.

## Configuración

Todo se configura en [params.yaml](params.yaml):

| Sección | Qué controla |
|---|---|
| `municipios` | Municipios incluidos (nombre en SIVIGILA y código DIVIPOLA). |
| `target` | Definición de brote: `metodo` (`bortman` o `percentil`), años de la ventana, semanas consecutivas y horizonte. |
| `particion` | Años de entrenamiento (hasta 2019), validación (2020–2022) y prueba (2023–2025). |
| `logistica`, `train`, `lstm` | Hiperparámetros de cada modelo. `train` es XGBoost. |
| `eventos` | Cómo se cuentan los brotes avisados. |
| `evaluate` | Bootstrap de los intervalos de confianza. |

**Las decisiones de diseño se toman solo con validación (2020–2022).** La prueba (2023–2025) se
consulta para el reporte final.

Para agregar un municipio:

1. Añádelo en `municipios`.
2. Genera su polígono con `src/geo/poligonos_mgn.py`.
3. Extrae su clima con `dvc repro --force clima@<municipio>`.
4. Corre `dvc repro`.

## Estructura

```
data/
  raw/                      Excel anuales de SIVIGILA (DVC)
  external/                 polígonos de cabeceras urbanas (DANE MGN 2021)
  processed/                SIVIGILA unificado y series del EDA (DVC)
  processed/municipios/<m>/ clima, cruce semanal y matriz de características (DVC)
models/<m>/                 modelos finales y modelos por año del walk-forward (DVC)
reports/
  figures/                  figuras del EDA
  municipios/<m>/           predicciones, métricas, coeficientes y evaluación
  resumen_municipios.csv    comparación de todos los municipios
src/
  rutas.py                  rutas por municipio y lectura de params.yaml
  geo/                      polígonos del MGN
  features/                 construcción de la matriz de características
  models/                   entrenamiento, evaluación y utilidades comunes (comun.py)
```

## Referencias

- Bortman, M. (1999). Elaboración de corredores o canales endémicos mediante planillas de cálculo.
  OPS. https://iris.paho.org/handle/10665.2/8562
- Umaña, J. D., Montenegro-Torres, J., Otero, J., et al. (2026). Endemic channel parametrization in
  dengue surveillance: methodological assessment of retrospective windows, outbreak trends, and
  zero-case periods in Colombia. *JMIR Public Health and Surveillance*, 12, e79914.
  https://doi.org/10.2196/79914
