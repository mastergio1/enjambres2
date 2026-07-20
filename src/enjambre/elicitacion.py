"""Elicitación de intención de compra a partir de la reacción en texto.

Punto clave de credibilidad (método SSR de PyMC Labs): NO se le pide al LLM
un número Likert directo —eso produce distribuciones irreales, con exceso de
respuestas neutras—. Se obtiene una reacción en lenguaje natural y luego se la
convierte a una escala 1-5.

Dos estrategias intercambiables, ambas exponen la misma interfaz
(``intencion`` + ``instruccion_prompt``) para el orquestador:

- ``ExtractorRating``: heurístico. El LLM declara el número; lo leemos. Simple,
  útil como línea base.
- ``SSR`` (Semantic Similarity Rating): el LLM solo reacciona; medimos la
  similitud semántica de su respuesta contra anclas Likert y calculamos la
  intención esperada. Es el método que sube la correlación con la realidad.

``SSR`` necesita un ``Embedder``. En dev usamos ``EmbedderLexico`` (offline,
determinista, sin dependencias). En producción se enchufa un embedder
semántico real (Voyage AI, sentence-transformers, etc.) sin tocar el resto.
"""
from __future__ import annotations

import math
import re
import unicodedata


def _tokenizar(texto: str) -> list[str]:
    desc = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(c for c in desc if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", sin_acentos)


# --------------------------------------------------------------------------- #
# Estrategia heurística
# --------------------------------------------------------------------------- #
class ExtractorRating:
    """Lee la intención de compra 1-5 que el propio LLM declara en el texto."""

    _PATRONES = (
        re.compile(r"intenci[oó]n de compra[:\s]*([1-5])\s*/\s*5", re.IGNORECASE),
        re.compile(r"\b([1-5])\s*/\s*5\b"),
    )

    def extraer(self, texto: str) -> int | None:
        for patron in self._PATRONES:
            m = patron.search(texto)
            if m:
                return int(m.group(1))
        return None

    def intencion(self, texto: str) -> float | None:
        v = self.extraer(texto)
        return float(v) if v is not None else None

    def instruccion_prompt(self) -> str:
        return (
            "Reacciona en 1-2 frases con tu opinión sincera y luego indica tu "
            "intención de compra en formato 'Intención de compra: X/5' "
            "(1 = jamás, 5 = seguro lo compro)."
        )


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder:
    """Interfaz de embeddings. Devuelve un vector por texto."""

    def embed(self, textos: list[str]) -> list[dict[str, float]]:
        raise NotImplementedError


class EmbedderLexico(Embedder):
    """Embedder offline determinista (bolsa de palabras TF normalizada).

    NO es semántico: sirve para desarrollo y tests sin dependencias ni llaves.
    En producción se reemplaza por embeddings reales; la interfaz no cambia.
    """

    def embed(self, textos: list[str]) -> list[dict[str, float]]:
        vecs: list[dict[str, float]] = []
        for t in textos:
            toks = _tokenizar(t)
            vec: dict[str, float] = {}
            if toks:
                for w in toks:
                    vec[w] = vec.get(w, 0.0) + 1.0 / len(toks)
            vecs.append(vec)
        return vecs


def _coseno(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    comun = set(a) & set(b)
    num = sum(a[w] * b[w] for w in comun)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Semantic Similarity Rating
# --------------------------------------------------------------------------- #
class SSR:
    """Convierte una reacción abierta en intención de compra por similitud.

    Mide la similitud de la respuesta contra cinco frases-ancla (una por punto
    de la escala), las convierte en una distribución con softmax y devuelve el
    valor esperado (1-5).
    """

    ANCLAS = {
        1: "No me interesa para nada, jamás lo compraría.",
        2: "No me convence, probablemente no lo compre.",
        3: "Tal vez, no estoy seguro, lo pensaría.",
        4: "Me gusta, probablemente lo compre.",
        5: "Me encanta, definitivamente lo compraría.",
    }

    def __init__(self, embedder: Embedder | None = None, nitidez: float = 10.0) -> None:
        self.embedder = embedder or EmbedderLexico()
        self.nitidez = nitidez
        self._niveles = sorted(self.ANCLAS)
        self._textos_ancla = [self.ANCLAS[n] for n in self._niveles]

    def distribucion(self, respuesta: str) -> dict[int, float]:
        if not respuesta or not respuesta.strip():
            return {}
        vecs = self.embedder.embed(self._textos_ancla + [respuesta])
        anclas, resp = vecs[:-1], vecs[-1]
        sims = [_coseno(resp, a) for a in anclas]
        pesos = [math.exp(self.nitidez * s) for s in sims]
        total = sum(pesos) or 1.0
        return {n: p / total for n, p in zip(self._niveles, pesos)}

    def intencion(self, respuesta: str) -> float | None:
        dist = self.distribucion(respuesta)
        if not dist:
            return None
        return sum(n * p for n, p in dist.items())

    # alias histórico
    def puntuar(self, respuesta: str) -> float | None:
        return self.intencion(respuesta)

    def instruccion_prompt(self) -> str:
        return (
            "Reacciona con tu opinión sincera en 1-3 frases: di qué te parece, "
            "si lo comprarías o no y por qué. Habla natural, no des un número."
        )
