"""Embedders semánticos reales (adaptadores) + fábrica.

Todos respetan la interfaz ``Embedder`` (``embed(textos) -> list[list[float]]``)
y se pueden enchufar al ``SSR`` sin tocar el resto del código. Los imports de
librerías externas están protegidos: instalar solo el que vayas a usar.

Selección:
    from enjambre.embeddings import crear_embedder
    emb = crear_embedder("st")       # SentenceTransformers (local, sin llave)
    emb = crear_embedder("openai")   # text-embedding-3-small (requiere OPENAI_API_KEY)
    emb = crear_embedder("voyage")   # voyage-3 (requiere VOYAGE_API_KEY)
    emb = crear_embedder("lexico")   # fallback offline determinista (no semántico)

O por entorno: ENJAMBRE_EMBEDDER=st
"""
from __future__ import annotations

import os

from .elicitacion import Embedder, EmbedderLexico


class EmbedderSentenceTransformers(Embedder):
    """Local, sin llaves. Multilingüe por defecto (bueno para español)."""

    nombre = "sentence-transformers"

    def __init__(
        self, modelo: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install sentence-transformers para usar este embedder."
            ) from exc
        self.modelo = modelo
        self._m = SentenceTransformer(modelo)

    def embed(self, textos: list[str]) -> list[list[float]]:  # pragma: no cover - requiere modelo
        vecs = self._m.encode(textos, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class EmbedderOpenAI(Embedder):
    """text-embedding-3-small (el sugerido en la literatura SSR). Requiere OPENAI_API_KEY."""

    nombre = "openai"

    def __init__(self, modelo: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai para usar este embedder.") from exc
        self.modelo = modelo
        self._cliente = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def embed(self, textos: list[str]) -> list[list[float]]:  # pragma: no cover - requiere API
        resp = self._cliente.embeddings.create(model=self.modelo, input=textos)
        return [d.embedding for d in resp.data]


class EmbedderVoyage(Embedder):
    """Voyage AI (el partner de embeddings recomendado por Anthropic). Requiere VOYAGE_API_KEY."""

    nombre = "voyage"

    def __init__(self, modelo: str = "voyage-3", api_key: str | None = None) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install voyageai para usar este embedder.") from exc
        self.modelo = modelo
        self._cliente = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    def embed(self, textos: list[str]) -> list[list[float]]:  # pragma: no cover - requiere API
        return self._cliente.embed(textos, model=self.modelo).embeddings


_REGISTRO = {
    "lexico": EmbedderLexico,
    "st": EmbedderSentenceTransformers,
    "sentence-transformers": EmbedderSentenceTransformers,
    "openai": EmbedderOpenAI,
    "voyage": EmbedderVoyage,
}


def crear_embedder(nombre: str | None = None) -> Embedder:
    """Crea un embedder por nombre (o desde ENJAMBRE_EMBEDDER; 'lexico' por defecto)."""
    nombre = (nombre or os.environ.get("ENJAMBRE_EMBEDDER", "lexico")).lower()
    if nombre not in _REGISTRO:
        raise ValueError(f"Embedder desconocido: {nombre}. Opciones: {sorted(_REGISTRO)}")
    return _REGISTRO[nombre]()
