"""Enjambre 2 — Motor de audiencias sintéticas para pre-test de productos (LATAM).

Producto núcleo de Rubicon Lab: subes un anuncio, concepto o mensaje y un
enjambre de consumidores sintéticos calibrados para mercados latinos
reacciona, rankea variantes y estima la intención de compra antes de lanzar.
"""
from .calibracion import RegistroCalibracion
from .elicitacion import SSR, ExtractorRating
from .enjambre import Enjambre, Reaccion, Resultado
from .llm import AnthropicClient, LLMClient, MockLLMClient
from .personas import GeneradorPersonas, Persona, Segmentacion

__version__ = "0.1.0"

__all__ = [
    "Enjambre",
    "Reaccion",
    "Resultado",
    "GeneradorPersonas",
    "Persona",
    "Segmentacion",
    "LLMClient",
    "MockLLMClient",
    "AnthropicClient",
    "ExtractorRating",
    "SSR",
    "RegistroCalibracion",
]
