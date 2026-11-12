"""
Extrae el polígono de la cabecera urbana (clase 1) de cada municipio del Marco Geoestadístico
Nacional del DANE (MGN 2021) y lo guarda simplificado como GeoJSON en data/external.

No forma parte del pipeline DVC porque el shapefile nacional pesa ~3 GB y no se versiona; los
GeoJSON resultantes sí se versionan en git. Uso:

    python src/geo/poligonos_mgn.py <carpeta con MGN_CLASE.shp/.shx/.dbf/.cpg>

Lee el shapefile sin dependencias externas (formato ESRI de polígonos + tabla dBase).
"""

import json
import struct
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SALIDA = ROOT_DIR / "data" / "external"
TOLERANCIA_GRADOS = 0.0005  # ~55 m: suficiente para píxeles de 250 m a 11 km


def leer_dbf(path, encoding):
    with open(path, "rb") as f:
        _, _, _, _, nrec, hlen, rlen = struct.unpack("<B3BIHH", f.read(12))
        f.seek(32)
        campos = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            campos.append((d[:11].split(b"\0")[0].decode(), d[16]))
        f.seek(hlen)
        for _ in range(nrec):
            r = f.read(rlen)
            pos, fila = 1, {}
            for nombre, largo in campos:
                fila[nombre] = r[pos:pos + largo].decode(encoding, "replace").strip()
                pos += largo
            yield fila


def leer_anillos(shp, shx, idx):
    shx.seek(100 + idx * 8)
    shp.seek(struct.unpack(">i", shx.read(4))[0] * 2 + 8 + 4 + 32)
    nparts, npts = struct.unpack("<2i", shp.read(8))
    partes = struct.unpack(f"<{nparts}i", shp.read(4 * nparts))
    pts = struct.unpack(f"<{2 * npts}d", shp.read(16 * npts))
    pts = list(zip(pts[0::2], pts[1::2]))
    return [pts[a:b] for a, b in zip(partes, list(partes[1:]) + [npts])]


def area(anillo):
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(anillo, anillo[1:] + anillo[:1]))


def douglas_peucker(puntos, tol):
    if len(puntos) < 3:
        return puntos
    (x1, y1), (x2, y2) = puntos[0], puntos[-1]
    dx, dy = x2 - x1, y2 - y1
    norma = (dx * dx + dy * dy) ** 0.5 or 1e-12
    distancias = [abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norma for x, y in puntos[1:-1]]
    i = max(range(len(distancias)), key=distancias.__getitem__)
    if distancias[i] <= tol:
        return [puntos[0], puntos[-1]]
    i += 1
    return douglas_peucker(puntos[:i + 1], tol)[:-1] + douglas_peucker(puntos[i:], tol)


def simplificar_anillo(anillo, tol):
    """Simplifica un anillo cerrado partiéndolo en el vértice más lejano al inicio
    (Douglas-Peucker sobre un anillo cerrado completo lo colapsa a 2 puntos)."""
    abierto = anillo[:-1] if anillo[0] == anillo[-1] else anillo
    x0, y0 = abierto[0]
    k = max(range(len(abierto)), key=lambda j: (abierto[j][0] - x0) ** 2 + (abierto[j][1] - y0) ** 2)
    mitad1 = douglas_peucker(abierto[:k + 1], tol)
    mitad2 = douglas_peucker(abierto[k:] + [abierto[0]], tol)
    simplificado = mitad1[:-1] + mitad2
    return [[round(x, 6), round(y, 6)] for x, y in simplificado]


def main(carpeta):
    with open(ROOT_DIR / "params.yaml", encoding="utf-8") as f:
        municipios = yaml.safe_load(f)["municipios"]
    por_codigo = {m["divipola"]: (slug, m) for slug, m in municipios.items()}

    encoding = (carpeta / "MGN_CLASE.cpg").read_text().strip() or "utf-8"
    filas = list(leer_dbf(carpeta / "MGN_CLASE.dbf", encoding))
    with open(carpeta / "MGN_CLASE.shp", "rb") as shp, open(carpeta / "MGN_CLASE.shx", "rb") as shx:
        for i, fila in enumerate(filas):
            if fila.get("CLAS_CCDGO") != "1" or fila.get("MPIO_CDPMP") not in por_codigo:
                continue
            slug, m = por_codigo[fila["MPIO_CDPMP"]]
            destino = SALIDA / f"{slug}_cabecera_urbana_mgn2021.geojson"
            if destino.exists():
                print(f"{slug}: ya existe {destino.name}, no se sobrescribe")
                continue
            # En el shapefile los anillos exteriores van en sentido horario (área negativa)
            exteriores = sorted((a for a in leer_anillos(shp, shx, i) if area(a) < 0), key=area)
            poligonos = [[simplificar_anillo(a, TOLERANCIA_GRADOS)] for a in exteriores]
            poligonos = [p for p in poligonos if len(p[0]) >= 4]
            geojson = {
                "type": "Feature",
                "properties": {
                    "municipio": m["nombre"], "MPIO_CDPMP": fila["MPIO_CDPMP"], "CLAS_CCDGO": "1",
                    "fuente": "DANE MGN 2021, cabecera municipal", "tolerancia_grados": TOLERANCIA_GRADOS,
                },
                "geometry": {"type": "MultiPolygon", "coordinates": poligonos},
            }
            destino.write_text(json.dumps(geojson), encoding="utf-8")
            print(f"{slug}: {len(poligonos)} polígono(s), {sum(len(p[0]) for p in poligonos)} vértices -> {destino.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
