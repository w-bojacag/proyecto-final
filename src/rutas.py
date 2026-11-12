"""
Rutas y parámetros por municipio, compartidos por todas las etapas del pipeline.

Cada municipio de params.yaml (sección `municipios`) tiene su propio clima, matrices, modelos y
reportes; los modelos de un municipio se entrenan solo con sus datos.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT_DIR / "params.yaml"
SIVIGILA_PATH = ROOT_DIR / "data" / "processed" / "dengue_bolivar_unificado.csv"


def cargar_params(seccion=None):
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get(seccion, {}) if seccion else params


@dataclass(frozen=True)
class Rutas:
    slug: str
    nombre: str
    divipola: str

    @property
    def poligono(self):
        return ROOT_DIR / "data" / "external" / f"{self.slug}_cabecera_urbana_mgn2021.geojson"

    @property
    def datos(self):
        return ROOT_DIR / "data" / "processed" / "municipios" / self.slug

    @property
    def clima(self):
        return self.datos / "clima_diario.csv"

    @property
    def cruce(self):
        return self.datos / "cruce_semanal.csv"

    @property
    def caracteristicas(self):
        return self.datos / "caracteristicas.csv"

    @property
    def modelos(self):
        return ROOT_DIR / "models" / self.slug

    @property
    def reportes(self):
        return ROOT_DIR / "reports" / "municipios" / self.slug

    def crear_carpetas(self):
        for carpeta in (self.datos, self.modelos, self.reportes):
            carpeta.mkdir(parents=True, exist_ok=True)
        return self


def rutas_municipio(slug):
    municipios = cargar_params("municipios")
    if slug not in municipios:
        raise SystemExit(f"Municipio '{slug}' no está en params.yaml. Opciones: {', '.join(municipios)}")
    m = municipios[slug]
    return Rutas(slug=slug, nombre=m["nombre"], divipola=str(m["divipola"])).crear_carpetas()


def municipio_desde_argumentos(descripcion):
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument("--municipio", required=True, help="clave del municipio en params.yaml")
    return rutas_municipio(parser.parse_args().municipio)
