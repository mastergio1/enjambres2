"""Elicitación de intención de compra a partir de texto libre.

Punto clave de credibilidad (PyMC Labs, método SSR): NO se le pide al LLM
un número Likert directo —eso produce distribuciones irreales, con exceso
de respuestas neutras—. Se obtiene una reacción en lenguaje natural y luego
se la convierte a una escala 1-5.

Para el MVP usamos ``ExtractorRating`` (heurístico: lee el número que el
propio modelo declara). ``SSR`` deja la interfaz del método completo por
similitud semántica, que se activará cuando enchufemos embeddings reales.
"""
from __future__ import annotations

import re


class ExtractorRating:
    """Extrae una intención de compra 1-5 del texto de la reacción."""

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


class SSR:
    """Semantic Similarity Rating (esqueleto).

    Convierte una respuesta abierta en una distribución sobre anclas
    Likert midiendo similitud semántica contra frases de referencia.
    Requiere un ``embedder`` (función texto -> vector). Se implementará en
    la Fase 2; por ahora documenta el contrato.
    """

    ANCLAS = {
        1: "No me interesa para nada, no lo compraría.",
        2: "Probablemente no lo compre.",
        3: "Tal vez, no estoy seguro.",
        4: "Probablemente lo compre.",
        5: "Definitivamente lo compraría.",
    }

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder

    def puntuar(self, respuesta_abierta: str) -> float:
        if self.embedder is None:
            raise NotImplementedError(
                "SSR requiere un embedder. Usa ExtractorRating en el MVP "
                "o enchufa embeddings en la Fase 2."
            )
        raise NotImplementedError  # pragma: no cover - Fase 2
