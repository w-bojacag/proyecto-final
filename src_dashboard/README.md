# Dashboard – alerta temprana de dengue (prototipo Streamlit)

Interfaz con cuatro vistas: casos y canal endémico con épocas de alto riesgo, predicción y alerta por fecha de corte, clima y detalle técnico del modelo. Consume la API de `../src_api`. Los datos y el modelo son **simulados**.

## Ejecución local (entorno virtual)

Primero debe estar activa la API en `localhost:8000` (ver [../src_api/README.md](../src_api/README.md)).

Windows (PowerShell):

```powershell
cd src_dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Linux / macOS / Git Bash:

```bash
cd src_dashboard
python -m venv .venv
source .venv/bin/activate        # en Git Bash en Windows: source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Abrir `http://localhost:8501`.

Variables de entorno: `API_URL` (por defecto `http://localhost:8000/api/v1`) y `API_TIMEOUT_S` (10).

## Vistas

1. **Casos y canal endémico**: casos semanales, umbral simulado y tramos de exceso sombreados.
2. **Predicción y alerta**: fecha de corte limitada a fechas con datos; muestra probabilidad, alerta y variables influyentes.
3. **Clima**: lluvia y temperatura semanales.
4. **Detalle del modelo**: nombre, versión, horizonte, variables y estado de métricas (pendientes).
