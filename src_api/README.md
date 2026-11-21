# API – alerta temprana de dengue (prototipo)

API FastAPI que expone el modelo de predicción de excesos de dengue. **Por ahora usa datos y modelo simulados** (`MODEL_BACKEND=simulado`); la interfaz `Predictor` en `app/modelo.py` permite cambiar al modelo real sin tocar los endpoints.

## Endpoints

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/v1/health` | Estado del servicio |
| GET | `/api/v1/municipios` | Lista de municipios |
| GET | `/api/v1/rango-fechas` | Fechas de corte válidas (las que tienen datos) |
| GET | `/api/v1/municipios/{id}/serie` | Casos, lluvia, temperatura, umbral del canal y exceso |
| POST | `/api/v1/prediccion` | Probabilidad de exceso a 4 semanas para una fecha de corte |
| GET | `/api/v1/modelo` | Información del modelo y estado de métricas |
| GET | `/api/v1/datos/estado` | Modo de datos activo y lo cargado en SQLite por municipio |
| POST | `/api/v1/datos/{municipio_id}/csv` | Carga `caracteristicas.csv` semanal en SQLite (actualiza las semanas existentes y agrega las nuevas) |

## Modos de datos

- **`MODO_DATOS=demo`** (por defecto): series simuladas. Todos los endpoints funcionan, incluida la predicción.
- **`MODO_DATOS=sqlite`**: `municipios/{id}/serie`, `rango-fechas` y `prediccion` leen de la base SQLite (`SQLITE_PATH`).

Los CSV se pueden cargar en cualquier modo; sólo se usan en `sqlite`.

### Formato: `caracteristicas.csv` del pipeline

Se carga el archivo `data/processed/municipios/<municipio>/caracteristicas.csv` tal cual lo genera el pipeline (una fila por año y semana epidemiológica). La API usa estas columnas y ignora el resto:

| Columna del CSV | Se guarda como | Significado |
|---|---|---|
| `ANO` | `anio` | Año epidemiológico |
| `SEMANA` | `semana` | Semana epidemiológica (1 a 53) |
| `casos_confirmados_o_probables` | `casos` | Casos de dengue de la semana (confirmados o probables) |
| `precipitation_mm` | `lluvia_mm` | Lluvia acumulada de la semana (mm) |
| `air_temp_mean_c` | `temp_media_c` | Temperatura media de la semana (°C) |
| `umbral_brote` | `umbral_canal` | Límite superior del canal endémico de Bortman |
| `target_brote_t4` (opcional) | `etiqueta` | Brote dentro de 4 semanas (0 o 1). Si no viene en el archivo, la gráfica no la muestra |

- `exceso` = `casos > umbral_canal`, igual que `es_brote_actual` en el pipeline.
- La fecha de inicio de cada semana (domingo) se calcula con el criterio INS/SIVIGILA.
- **Carga por actualización:** la clave es `(municipio, ANO, SEMANA)`. Si la semana ya existe se **actualiza**; si no existe se **agrega**; las semanas que no vienen en el archivo se **conservan**. La respuesta indica cuántas fueron nuevas y cuántas actualizadas.
- Si faltan columnas, hay valores vacíos o no numéricos, o hay semanas repetidas, la API responde `422` con el detalle.

Ejemplo:

```bash
curl -F "archivo=@data/processed/municipios/cartagena/caracteristicas.csv" http://localhost:8000/api/v1/datos/cartagena/csv
```

Ejemplo de predicción:

```json
POST /api/v1/prediccion
{"municipio_id": "cartagena", "fecha_corte": "2024-06-02"}
```

Documentación interactiva en `http://localhost:8000/docs`.

## Ejecución local (entorno virtual)

Windows (PowerShell):

```powershell
cd src_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Linux / macOS / Git Bash:

```bash
cd src_api
python -m venv .venv
source .venv/bin/activate        # en Git Bash en Windows: source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Verificación rápida: `http://localhost:8000/api/v1/health` debe responder `{"estado": "ok", ...}`.

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `MODO_DATOS` | `demo` | Fuente de las series: `demo` (simuladas) o `sqlite` (CSV cargados) |
| `SQLITE_PATH` | `src_api/data/dengue.sqlite` | Archivo de la base SQLite (se crea al cargar el primer CSV) |
| `MODEL_BACKEND` | `simulado` | Backend del modelo (ver `app/modelo.py`) |
| `MODEL_VERSION` | `simulado-0.1` | Versión reportada en cada respuesta |
| `HORIZONTE_SEMANAS` | `4` | Semanas de anticipación |
| `UMBRAL_DECISION` | `0.5` | Probabilidad a partir de la cual hay alerta (pendiente de validar) |
| `ANIO_ENTRENAMIENTO_FIN` | `2019` | Fin del periodo de entrenamiento (propuesta: 2007–2019) |
| `ANIO_MINIMO_GRAFICOS` | `2012` | Primer año que devuelve `municipios/{id}/serie` (gráficas del canal endémico y clima). Los años anteriores se omiten; la predicción no se ve afectada |
| `SEMILLA_SIMULACION` | `42` | Semilla de los datos simulados |

## Cómo conectar el modelo real

1. Implementar una clase que herede de `Predictor` en `app/modelo.py`.
2. Cargar el artefacto guardado y construir las variables igual que en entrenamiento, usando sólo el historial hasta `fecha_corte`.
3. Registrarla en `cargar_predictor()` con un nuevo valor de `MODEL_BACKEND`.
4. Reemplazar `_serie_completa()` en `app/datos.py` por la lectura de los datos procesados.

## Pendientes

- Datos reales (SIVIGILA, Google Earth Engine) y canal endémico definitivo.
- Modelo entrenado y métricas de evaluación (sin métricas por ahora).
- Umbral de decisión validado con el equipo.
