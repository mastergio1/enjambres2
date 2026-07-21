"""Adaptadores de fuentes externas al corpus RAG.

Convierten datos crudos (p. ej. opiniones extraídas de X, organizadas por país y
producto) en ``Documento`` que ``BaseConocimiento`` puede consultar. Se mantiene
aparte del núcleo para que agregar fuentes nuevas no toque la recuperación.

Nota de diseño: NO ponderamos por engagement. En estos datos el mayor engagement
suele ser un meme o un chiste; ponderar por popularidad amplificaría el ruido en
vez de la señal. El engagement se guarda como metadato de trazabilidad, no como
peso de recuperación.
"""
from __future__ import annotations

import json
from pathlib import Path

from .conocimiento import BaseConocimiento, Documento

# Normaliza el país al mismo valor que usan las personas (con acentos).
_PAISES = {
    "mexico": "México", "méxico": "México",
    "argentina": "Argentina", "chile": "Chile", "colombia": "Colombia",
    "peru": "Perú", "perú": "Perú", "ecuador": "Ecuador",
    "venezuela": "Venezuela", "bolivia": "Bolivia", "guatemala": "Guatemala",
    "rep. dominicana": "Rep. Dominicana",
}


def _norm_pais(pais: str) -> str:
    return _PAISES.get((pais or "").strip().lower(), pais)


# Sentimientos que son memes/chistes, no opinión de producto: se excluyen del
# corpus de DECISIÓN, pero no se borran (quedan separables y contables).
def es_meme(doc: Documento) -> bool:
    return (doc.tags.get("sentimiento") or "").lower().startswith("humor")


def particionar_memes(docs: list[Documento]) -> tuple[list[Documento], list[Documento]]:
    """Separa (opiniones_serias, memes) sin descartar nada."""
    serios = [d for d in docs if not es_meme(d)]
    memes = [d for d in docs if es_meme(d)]
    return serios, memes


def opiniones_x_a_documentos(datos: list[dict]) -> list[Documento]:
    """Aplana la estructura país → producto → opiniones a una lista de Documentos."""
    docs: list[Documento] = []
    for bloque in datos:
        pais = _norm_pais(bloque.get("pais", ""))
        for producto, opiniones in bloque.get("productos", {}).items():
            for op in opiniones:
                texto = (op.get("text") or "").strip()
                if not texto:
                    continue
                tags = {"pais": pais, "producto": producto}
                if op.get("sentimiento"):
                    tags["sentimiento"] = str(op["sentimiento"])
                if op.get("engagement") is not None:
                    tags["engagement"] = str(op["engagement"])  # trazabilidad, no peso
                docs.append(Documento(texto=texto, tags=tags, fuente="X"))
    return docs


def cargar_opiniones_x(ruta: str | Path, excluir_memes: bool = True) -> BaseConocimiento:
    """Carga opiniones de X como corpus RAG.

    Por defecto excluye los memes/chistes (sentimiento ``humor*``) del corpus de
    decisión. La base recuerda cuántos excluyó en ``base.memes_excluidos`` para
    poder reportarlo (no se ocultan, se separan).
    """
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    docs = opiniones_x_a_documentos(datos)
    serios, memes = particionar_memes(docs)
    base = BaseConocimiento(serios if excluir_memes else docs)
    base.memes_excluidos = len(memes) if excluir_memes else 0
    base.memes = memes  # disponibles aparte, nunca borrados
    return base
