"""Base de conocimiento para aterrizar personas en datos reales (RAG).

Sin esto, las personas opinan desde el vacío del pre-entrenamiento. Con
esto, cada persona reacciona teniendo a la vista lo que gente parecida ha
dicho de verdad (reseñas, tickets de soporte, social listening del cliente).

El MVP usa un recuperador TF-IDF en stdlib (sin dependencias, insensible a
acentos). ``Recuperador`` es la interfaz para enchufar embeddings en el
futuro sin tocar el resto.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def _tokenizar(texto: str) -> list[str]:
    desc = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(c for c in desc if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", sin_acentos)


@dataclass
class Documento:
    texto: str
    tags: dict[str, str] = field(default_factory=dict)
    fuente: str = ""


class Recuperador:
    """Interfaz de recuperación (swappable por embeddings más adelante)."""

    def recuperar(
        self, consulta: str, *, k: int = 3, tags_preferidos: dict[str, str] | None = None
    ) -> list[Documento]:
        raise NotImplementedError


class RecuperadorTFIDF(Recuperador):
    """Recuperador TF-IDF con un pequeño boost por coincidencia de tags."""

    def __init__(self, documentos: list[Documento]) -> None:
        self.documentos = documentos
        self._tokens = [_tokenizar(d.texto) for d in documentos]
        n = len(documentos) or 1
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        self._idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self._doc_vecs = [self._vector(toks) for toks in self._tokens]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {w: (c / total) * self._idf.get(w, 1.0) for w, c in tf.items()}

    def recuperar(
        self, consulta: str, *, k: int = 3, tags_preferidos: dict[str, str] | None = None
    ) -> list[Documento]:
        q = self._vector(_tokenizar(consulta))
        if not q:
            return []
        puntuados: list[tuple[float, Documento]] = []
        for doc, vec in zip(self.documentos, self._doc_vecs):
            score = sum(peso * vec.get(w, 0.0) for w, peso in q.items())
            if tags_preferidos:
                coincidencias = sum(1 for kk, vv in tags_preferidos.items() if doc.tags.get(kk) == vv)
                score *= 1 + 0.4 * coincidencias
            if score > 0:
                puntuados.append((score, doc))
        puntuados.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in puntuados[:k]]


class BaseConocimiento:
    """Colección de documentos consultable por una persona/estímulo."""

    def __init__(
        self, documentos: list[Documento] | None = None, recuperador: Recuperador | None = None
    ) -> None:
        self.documentos = list(documentos or [])
        self.recuperador = recuperador or RecuperadorTFIDF(self.documentos)

    @classmethod
    def desde_json(cls, ruta: Path | str) -> "BaseConocimiento":
        data = json.loads(Path(ruta).read_text(encoding="utf-8"))
        docs = [
            Documento(texto=d["texto"], tags=d.get("tags", {}), fuente=d.get("fuente", ""))
            for d in data
        ]
        return cls(docs)

    def recuperar(
        self, consulta: str, *, k: int = 3, tags_preferidos: dict[str, str] | None = None
    ) -> list[Documento]:
        if not self.documentos:
            return []
        return self.recuperador.recuperar(consulta, k=k, tags_preferidos=tags_preferidos)
