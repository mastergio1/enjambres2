"""Lógica de la interfaz "sube y reacciona" (independiente del transporte HTTP).

Un no-técnico pega un producto y sus variantes, elige la audiencia, y recibe la
reacción del enjambre + el ranking. Esta capa arma el enjambre según los
parámetros y devuelve un dict listo para serializar. El servidor (servidor.py)
solo la expone por HTTP; así la lógica se puede testear sin levantar un puerto.
"""
from __future__ import annotations

import os
from pathlib import Path

from .backtest import GeneradorChile
from .elicitacion import ANCLAS_CHILE, SSR, ExtractorRating
from .enjambre import Enjambre
from .fuentes import cargar_opiniones_x
from .llm import MockLLMClient
from .personas import GeneradorPersonas, Segmentacion

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"

_corpus = None  # corpus RAG cargado una sola vez


def _conocimiento():
    """Corpus de opiniones reales (X). Se carga perezosamente; None si no existe."""
    global _corpus
    if _corpus is None:
        ruta = DATOS / "opiniones_x_por_pais.json"
        _corpus = cargar_opiniones_x(ruta) if ruta.exists() else False
    return _corpus or None

MERCADOS = {
    "latam": "LATAM (general, español)",
    "chile": "Chile (Censo 2024 + GSE, solo AB→C2)",
}
METODOS = {
    "rapido": "Rápido — el agente declara su intención",
    "semantico": "Semántico (SSR) — mide por similitud, evita el sesgo del número",
}
N_MAX = 200


def opciones() -> dict:
    return {
        "mercados": MERCADOS,
        "metodos": METODOS,
        "n_personas_default": 40,
        "n_personas_max": N_MAX,
        "llm": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "simulado",
    }


def _generador(mercado: str, semilla: int = 7):
    if mercado == "chile":
        ruta = DATOS / "segmentos_chile.example.json"
        if ruta.exists():
            return GeneradorChile.desde_json(ruta, semilla=semilla)
    ruta = DATOS / "segmentos_latam.example.json"
    seg = Segmentacion.desde_json(ruta) if ruta.exists() else Segmentacion()
    return GeneradorPersonas(seg, semilla=semilla)


def _elicitador(metodo: str):
    if metodo == "semantico":
        return SSR(anclas=ANCLAS_CHILE)
    return ExtractorRating()


def _llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .llm import AnthropicClient

        return AnthropicClient(modelo="claude-sonnet-5")
    return MockLLMClient()


def _normalizar_variantes(bruto) -> dict[str, str]:
    variantes: dict[str, str] = {}
    if isinstance(bruto, list):
        for i, v in enumerate(bruto, 1):
            nombre = (v.get("nombre") or f"Variante {i}").strip()
            texto = (v.get("texto") or "").strip()
            if texto:
                variantes[nombre or f"Variante {i}"] = texto
    elif isinstance(bruto, dict):
        variantes = {k: v.strip() for k, v in bruto.items() if (v or "").strip()}
    return variantes


def ejecutar_pretest(payload: dict) -> dict:
    """Corre el pre-test. Lanza ValueError con un mensaje claro si falta algo."""
    producto = (payload.get("producto") or "").strip()
    if not producto:
        raise ValueError("Falta la descripción del producto.")
    variantes = _normalizar_variantes(payload.get("variantes"))
    if not variantes:
        raise ValueError("Agrega al menos una variante con texto.")

    try:
        n = int(payload.get("n_personas", 40))
    except (TypeError, ValueError):
        n = 40
    n = max(1, min(n, N_MAX))

    mercado = payload.get("mercado", "latam")
    if mercado not in MERCADOS:
        mercado = "latam"
    metodo = payload.get("metodo", "rapido")
    if metodo not in METODOS:
        metodo = "rapido"
    contexto = (payload.get("contexto") or "").strip()

    enjambre = Enjambre(
        llm=_llm(), generador=_generador(mercado), elicitador=_elicitador(metodo),
        conocimiento=_conocimiento(),
    )
    ctx = f"{producto}. {contexto}".strip().strip(".")
    ranking = enjambre.comparar(variantes, n_personas=n, contexto=ctx)

    resultados = []
    for nombre, res in ranking:
        resultados.append({
            "nombre": nombre,
            "media": round(res.media(), 2),
            "top_2_box": round(res.top_2_box(), 1),
            "dispersion": round(res.dispersion(), 2),
            "distribucion": res.distribucion(),
            "muestras": [r.texto.strip() for r in res.reacciones[:3]],
        })

    # Cobertura RAG: nunca en silencio (Tarea 7).
    cobertura = ranking[0][1].cobertura_rag if ranking else None
    advertencias = []
    if cobertura and cobertura.get("advertencia"):
        advertencias.append(cobertura["advertencia"])

    return {
        "producto": producto,
        "contexto": contexto,
        "mercado": mercado,
        "metodo": metodo,
        "n_personas": n,
        "llm": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "simulado",
        "ganadora": resultados[0]["nombre"] if resultados else None,
        "resultados": resultados,
        "cobertura_rag": cobertura,
        "advertencias": advertencias,
    }
