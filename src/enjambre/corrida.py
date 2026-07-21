"""Tipo de corrida: deducido del entorno, nunca declarado a mano.

Distingue dos cosas que se parecen pero significan lo opuesto:

- ``VALIDACION_ARNES``: prueba que el instrumento detecta señal cuando la hay
  (mide el termómetro). NO es desempeño del producto.
- ``CALIBRACION_REAL``: el enjambre contra productos reales con notas reales
  (mide la fiebre). Este sí es el número que va a un cliente.

El tipo se DEDUCE de condiciones verificables del entorno (qué LLM, qué
dataset), no de lo que declare quien corre. Si alguien declara un tipo que
contradice lo deducido, se ignora el parámetro y se avisa qué condición falló.
"""
from __future__ import annotations

from enum import Enum


class TipoCorrida(str, Enum):
    VALIDACION_ARNES = "VALIDACION_ARNES"
    CALIBRACION_REAL = "CALIBRACION_REAL"


def condiciones_realidad(llm, dataset) -> list[dict]:
    """Devuelve las condiciones que deben cumplirse TODAS para ser calibración real."""
    llm_real = bool(getattr(llm, "es_real", False))
    ds_real = bool(getattr(dataset, "es_real", False))
    return [
        {
            "condicion": "llm_real",
            "cumple": llm_real,
            "detalle": (
                f"el cliente LLM ({type(llm).__name__}) está marcado como real"
                if llm_real
                else f"el cliente LLM ({type(llm).__name__}) no es un LLM real (es simulación)"
            ),
        },
        {
            "condicion": "dataset_real",
            "cumple": ds_real,
            "detalle": (
                "el dataset es ciego, con procedencia externa no sintética"
                if ds_real
                else "el dataset no es real (es de ejemplo, sin procedencia externa, o sintético)"
            ),
        },
    ]


def deducir_tipo(llm, dataset, tipo_declarado: TipoCorrida | str | None = None) -> dict:
    """Deduce el tipo de corrida. Devuelve tipo, condiciones y avisos.

    Reglas:
    - Si el LLM no es real  → VALIDACION_ARNES (sin excepción).
    - Si el dataset no es real → VALIDACION_ARNES (sin excepción).
    - Solo si TODAS las condiciones de realidad se cumplen → CALIBRACION_REAL.
    - Si ``tipo_declarado`` contradice lo deducido, se ignora y se avisa.
    """
    condiciones = condiciones_realidad(llm, dataset)
    tipo = (
        TipoCorrida.CALIBRACION_REAL
        if all(c["cumple"] for c in condiciones)
        else TipoCorrida.VALIDACION_ARNES
    )

    avisos: list[str] = []
    if tipo_declarado is not None:
        declarado = TipoCorrida(tipo_declarado) if not isinstance(tipo_declarado, TipoCorrida) else tipo_declarado
        if declarado != tipo:
            faltantes = [c["detalle"] for c in condiciones if not c["cumple"]]
            avisos.append(
                f"Se ignoró el tipo declarado '{declarado.value}': el entorno indica "
                f"'{tipo.value}'. Condición(es) no cumplida(s): " + "; ".join(faltantes)
                if faltantes
                else f"Se ignoró el tipo declarado '{declarado.value}': el entorno indica '{tipo.value}'."
            )

    return {"tipo": tipo, "condiciones": condiciones, "avisos": avisos}
