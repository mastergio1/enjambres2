"""Entrada del dataset de calibración: estímulo ciego separado del resultado.

Diseño central (Tarea 6): el objeto que se le entrega al enjambre es de un tipo
—``EstimuloCiego``— que **estructuralmente no tiene** campo de nota. No es un
objeto con el campo vacío (que se podría llenar por accidente): es un tipo que
no tiene dónde poner el resultado. El resultado real vive aparte, en la
``ClaveResultado``, y solo se junta con las predicciones en el momento del
scoring, nunca antes.

Por qué importa: si el resultado se filtra al estímulo, la correlación sale
altísima y parece que el producto funciona. Es el fallo más peligroso porque es
silencioso: nada se rompe, solo salen números buenos y falsos.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Campos que NO pueden aparecer en los estímulos ciegos (delatarían la nota).
CAMPOS_PROHIBIDOS = {
    "nota", "nota_real", "valoracion", "valoracion_real", "rating", "score",
    "estrellas", "stars", "puntaje", "calificacion", "review_score", "promedio",
    "n_resenas", "num_reviews", "ranking",
}

# Fuentes que se consideran sintéticas → nunca habilitan CALIBRACION_REAL.
_RE_SINTETICA = re.compile(r"ejemplo|sint[eé]tic|demo|prueba|test|mock", re.IGNORECASE)


def _sintetica(fuente: str) -> bool:
    return bool(_RE_SINTETICA.search(fuente or ""))


def _sha1(ruta: Path) -> str:
    return hashlib.sha1(Path(ruta).read_bytes()).hexdigest()[:12]


@dataclass(frozen=True)
class EstimuloCiego:
    """Lo único que ve el enjambre. Sin nota: no hay dónde ponerla."""

    id: str
    tipo: str
    nombre: str
    categoria: str
    descripcion: str

    def texto(self) -> str:
        return (
            f"ESTÍMULO — Mira esta {self.tipo} y reacciona como lo harías tú:\n\n"
            f"Nombre: {self.nombre}\n"
            f"Categoría: {self.categoria}\n"
            f"Descripción:\n{self.descripcion}"
        )


@dataclass(frozen=True)
class Procedencia:
    fuente: str
    fecha: str
    n_muestras: int | None = None


@dataclass(frozen=True)
class ClaveResultado:
    id: str
    valoracion_real: float
    procedencia: Procedencia


@dataclass
class DatasetCalibracion:
    """Estímulos ciegos + clave de resultados, mantenidos separados.

    El enjambre solo recibe ``estimulos``. La clave (``_clave``) se consulta
    únicamente en el scoring. ``es_real`` se DEDUCE: verdadero solo si el dataset
    es ciego, con procedencia presente y no sintética.
    """

    estimulos: list[EstimuloCiego]
    origen: str
    es_real: bool
    sha1_estimulos: str
    sha1_clave: str
    _clave: dict[str, ClaveResultado]

    def resultado(self, id_estimulo: str) -> float:
        return self._clave[id_estimulo].valoracion_real

    def procedencia(self, id_estimulo: str) -> Procedencia:
        return self._clave[id_estimulo].procedencia

    def notas(self) -> dict[str, float]:
        return {k: v.valoracion_real for k, v in self._clave.items()}

    def n_muestras(self, id_estimulo: str) -> int | None:
        return self._clave[id_estimulo].procedencia.n_muestras


def cargar_dataset_ciego(estimulos_path: str | Path, clave_path: str | Path) -> DatasetCalibracion:
    """Carga y VALIDA un dataset real de dos archivos separados.

    Rechaza con error si: los estímulos contienen algún campo de nota; los IDs
    de ambos archivos no coinciden exactamente; o algún ítem de la clave no trae
    procedencia (fuente y fecha).
    """
    est_raw = json.loads(Path(estimulos_path).read_text(encoding="utf-8"))
    clave_raw = json.loads(Path(clave_path).read_text(encoding="utf-8"))

    # (b1) Ningún campo de nota en los estímulos.
    for d in est_raw:
        contaminados = set(d) & CAMPOS_PROHIBIDOS
        if contaminados:
            raise ValueError(
                f"'estimulos_ciegos' contiene campos de resultado prohibidos "
                f"{sorted(contaminados)} en el ítem '{d.get('id')}'. Los estímulos "
                "deben ser ciegos: la nota va solo en la clave de resultados."
            )
    estimulos = [
        EstimuloCiego(
            id=d["id"], tipo=d.get("tipo", "item"), nombre=d["nombre"],
            categoria=d.get("categoria", ""), descripcion=d["descripcion"],
        )
        for d in est_raw
    ]

    # (b3) Procedencia obligatoria en cada resultado.
    clave: dict[str, ClaveResultado] = {}
    fuentes: list[str] = []
    for d in clave_raw:
        fuente = (d.get("fuente") or "").strip()
        fecha = (d.get("fecha") or "").strip()
        if not fuente or not fecha:
            raise ValueError(
                f"El ítem '{d.get('id')}' de la clave no tiene procedencia completa "
                "(se requieren 'fuente' y 'fecha')."
            )
        fuentes.append(fuente)
        clave[d["id"]] = ClaveResultado(
            id=d["id"],
            valoracion_real=float(d["valoracion_real"]),
            procedencia=Procedencia(fuente=fuente, fecha=fecha, n_muestras=d.get("n_muestras")),
        )

    # (b2) IDs coinciden exactamente.
    ids_est, ids_clave = {e.id for e in estimulos}, set(clave)
    if ids_est != ids_clave:
        raise ValueError(
            "Los IDs de estímulos y clave no coinciden. "
            f"Solo en estímulos: {sorted(ids_est - ids_clave)}; "
            f"solo en clave: {sorted(ids_clave - ids_est)}."
        )

    es_real = len(fuentes) > 0 and not any(_sintetica(f) for f in fuentes)
    return DatasetCalibracion(
        estimulos=estimulos,
        origen="externo" if es_real else "externo-sintetico",
        es_real=es_real,
        sha1_estimulos=_sha1(Path(estimulos_path)),
        sha1_clave=_sha1(Path(clave_path)),
        _clave=clave,
    )


def cargar_dataset_ejemplo(items_path: str | Path) -> DatasetCalibracion:
    """Carga el dataset de ejemplo (notas inline) como ciego y NO real.

    El archivo de ejemplo trae la nota junto al estímulo; aquí la separamos y
    marcamos el dataset como sintético (es_real=False), de modo que jamás pueda
    deducirse como calibración real.
    """
    data = json.loads(Path(items_path).read_text(encoding="utf-8"))
    estimulos = [
        EstimuloCiego(
            id=d["id"], tipo=d.get("tipo", "item"), nombre=d["nombre"],
            categoria=d.get("categoria", ""), descripcion=d["descripcion"],
        )
        for d in data
    ]
    clave = {
        d["id"]: ClaveResultado(
            id=d["id"], valoracion_real=float(d["nota_real"]),
            procedencia=Procedencia(fuente="ejemplo-sintetico", fecha="2026-07-21",
                                     n_muestras=d.get("n_resenas")),
        )
        for d in data
    }
    h = _sha1(Path(items_path))
    return DatasetCalibracion(
        estimulos=estimulos, origen="ejemplo", es_real=False,
        sha1_estimulos=h, sha1_clave=h, _clave=clave,
    )


# --------------------------------------------------------------------------- #
# Selección estratificada (usa la clave, del lado del operador — nunca el agente)
# --------------------------------------------------------------------------- #
PROPORCION_BANDAS = {"malo": 0.20, "medio": 0.30, "bueno": 0.30, "excelente": 0.20}


def banda(nota: float) -> str:
    if nota <= 3.0:
        return "malo"
    if nota <= 3.9:
        return "medio"
    if nota <= 4.4:
        return "bueno"
    return "excelente"


def seleccion_estratificada(
    dataset: DatasetCalibracion, n: int = 50, min_muestras: int = 50, semilla: int | None = 0
) -> list[EstimuloCiego]:
    """Reparte por todo el rango de nota real para fabricar dispersión.

    Usa la clave (lado del operador) para elegir; devuelve solo estímulos ciegos.
    """
    import random

    rng = random.Random(semilla)
    notas = dataset.notas()
    candidatos = [
        e for e in dataset.estimulos if (dataset.n_muestras(e.id) or 0) >= min_muestras
    ]
    por_banda: dict[str, list[EstimuloCiego]] = {b: [] for b in PROPORCION_BANDAS}
    for e in candidatos:
        por_banda[banda(notas[e.id])].append(e)
    seleccion: list[EstimuloCiego] = []
    for b, prop in PROPORCION_BANDAS.items():
        grupo = por_banda[b][:]
        rng.shuffle(grupo)
        seleccion.extend(grupo[: max(0, round(prop * n))])
    return seleccion
