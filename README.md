# Sistema de alerta temprana de dengue – Bolívar (prototipo)

Componentes:

- `src_api/`: API FastAPI (modelo simulado por ahora). Ver [src_api/README.md](src_api/README.md).
- `src_dashboard/`: interfaz Streamlit. Ver [src_dashboard/README.md](src_dashboard/README.md).
- `docker-compose.yml`: levanta ambos servicios.

## Ejecutar con Docker

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API y documentación: http://localhost:8000/docs

## Ejecutar en local (entornos virtuales)

Requiere Python 3.12. Usar dos terminales, una por componente.

**Terminal 1: API** (desde `proyecto-final/`)

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

**Terminal 2: dashboard** (desde `proyecto-final/`, con la API ya activa)

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

Enlaces:

- Dashboard: http://localhost:8501
- API y documentación: http://localhost:8000/docs

Si el dashboard no conecta con la API, revise que la terminal 1 siga corriendo y que `API_URL` apunte a `http://localhost:8000/api/v1` (valor por defecto).
