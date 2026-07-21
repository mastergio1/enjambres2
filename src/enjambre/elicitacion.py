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

import hashlib
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
    """Interfaz de embeddings: devuelve un vector denso (list[float]) por texto.

    Los adaptadores reales (SentenceTransformers, OpenAI, Voyage) viven en
    ``enjambre.embeddings`` y respetan esta misma interfaz.
    """

    def embed(self, textos: list[str]) -> list[list[float]]:
        raise NotImplementedError


class EmbedderLexico(Embedder):
    """Embedder offline determinista: bolsa de palabras con *feature hashing* a
    un vector denso L2-normalizado.

    NO es semántico (no distingue "me sirve" de "no me sirve"): sirve para
    desarrollo y tests sin dependencias ni llaves. En producción se reemplaza
    por un embedder semántico real (``enjambre.embeddings.crear_embedder``); la
    interfaz no cambia.
    """

    nombre = "lexico"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _rasgo(self, palabra: str) -> tuple[int, float]:
        h = int(hashlib.md5(palabra.encode("utf-8")).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 17) & 1 else -1.0)

    def embed(self, textos: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in textos:
            v = [0.0] * self.dim
            for w in _tokenizar(t):
                idx, signo = self._rasgo(w)
                v[idx] += signo
            norma = math.sqrt(sum(x * x for x in v))
            vecs.append([x / norma for x in v] if norma else v)
        return vecs


def _coseno(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Semantic Similarity Rating
# --------------------------------------------------------------------------- #
# Juego de anclas por defecto.
ANCLAS_DEFAULT = {
    1: "No me interesa para nada, jamás lo compraría.",
    2: "No me convence, probablemente no lo compre.",
    3: "Tal vez, no estoy seguro, lo pensaría.",
    4: "Me gusta, probablemente lo compre.",
    5: "Me encanta, definitivamente lo compraría.",
}

# Anclas del paquete de backtest Chile (español de Chile, intención de compra/descarga).
ANCLAS_CHILE = {
    1: "De ninguna manera compraría o descargaría esto; no me interesa para nada.",
    2: "Probablemente no lo compraría; no me convence y le veo problemas.",
    3: "Me da lo mismo; podría o no comprarlo, no me genera nada especial.",
    4: "Probablemente sí lo compraría o descargaría; me parece bueno y útil.",
    5: "Sin duda lo compraría o descargaría; me encanta y es justo lo que busco.",
}


class SSR:
    """Convierte una reacción abierta en intención de compra por similitud.

    Mide la similitud de la respuesta contra cinco frases-ancla (una por punto
    de la escala), las convierte en una distribución con softmax y devuelve el
    valor esperado (1-5).

    ``anclas`` puede ser un juego (dict {1..5: texto}) o una lista de juegos; en
    ese caso se promedian las distribuciones, como recomienda el método SSR para
    reducir la sensibilidad a la redacción de las anclas.
    """

    ANCLAS = ANCLAS_DEFAULT

    def __init__(
        self,
        embedder: Embedder | None = None,
        nitidez: float = 10.0,
        anclas: dict | list[dict] | None = None,
    ) -> None:
        self.embedder = embedder or EmbedderLexico()
        self.nitidez = nitidez
        if anclas is None:
            conjuntos = [self.ANCLAS]
        elif isinstance(anclas, dict):
            conjuntos = [anclas]
        else:
            conjuntos = list(anclas)
        # Precalcula (cachea) los embeddings de las anclas una sola vez: por cada
        # reacción solo se embebe la respuesta, no las anclas.
        self._conjuntos = [
            (sorted(c), self.embedder.embed([c[n] for n in sorted(c)])) for c in conjuntos
        ]

    def distribucion(self, respuesta: str) -> dict[int, float]:
        if not respuesta or not respuesta.strip():
            return {}
        resp = self.embedder.embed([respuesta])[0]
        acumulado: dict[int, float] = {}
        for niveles, anclas_emb in self._conjuntos:
            sims = [_coseno(resp, a) for a in anclas_emb]
            pesos = [math.exp(self.nitidez * s) for s in sims]
            total = sum(pesos) or 1.0
            for n, p in zip(niveles, pesos):
                acumulado[n] = acumulado.get(n, 0.0) + p / total
        k = len(self._conjuntos)
        return {n: v / k for n, v in acumulado.items()}

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
