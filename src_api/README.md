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

Ejemplo:

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
| `MODEL_BACKEND` | `simulado` | Backend del modelo (ver `app/modelo.py`) |
| `MODEL_VERSION` | `simulado-0.1` | Versión reportada en cada respuesta |
| `HORIZONTE_SEMANAS` | `4` | Semanas de anticipación |
| `UMBRAL_DECISION` | `0.5` | Probabilidad a partir de la cual hay alerta (pendiente de validar) |
| `ANIO_ENTRENAMIENTO_FIN` | `2019` | Fin del periodo de entrenamiento (propuesta: 2007–2019) |
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
