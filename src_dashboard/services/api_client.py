"""Cliente HTTP de la API. Sin dependencias de Streamlit."""
import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1").rstrip("/")
TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "10"))


class ApiError(Exception):
    """Error de conexión o de validación devuelto por la API."""


def _request(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_S, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(
            f"No se pudo conectar con la API en {API_URL}. Verifique que esté en ejecución."
        ) from exc
    if r.status_code >= 400:
        try:
            detalle = r.json().get("detail", r.text)
        except ValueError:
            detalle = r.text
        raise ApiError(str(detalle))
    return r.json()


def get_municipios() -> list[dict]:
    return _request("GET", "/municipios")


def get_rango_fechas() -> dict:
    return _request("GET", "/rango-fechas")


def get_serie(municipio_id: str) -> list[dict]:
    return _request("GET", f"/municipios/{municipio_id}/serie")


def post_prediccion(municipio_id: str, fecha_corte: str) -> dict:
    return _request(
        "POST",
        "/prediccion",
        json={"municipio_id": municipio_id, "fecha_corte": fecha_corte},
    )


def get_modelo() -> dict:
    return _request("GET", "/modelo")


def get_estado_datos() -> dict:
    return _request("GET", "/datos/estado")


def post_csv(municipio_id: str, nombre_archivo: str, contenido: bytes) -> dict:
    return _request(
        "POST",
        f"/datos/{municipio_id}/csv",
        files={"archivo": (nombre_archivo, contenido, "text/csv")},
    )
