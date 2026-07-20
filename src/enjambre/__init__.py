"""Enjambre 2 — Motor de audiencias sintéticas para pre-test de productos (LATAM).

Producto núcleo de Rubicon Lab: subes un anuncio, concepto o mensaje y un
enjambre de consumidores sintéticos calibrados para mercados latinos
reacciona, rankea variantes y estima la intención de compra antes de lanzar.
"""
from . import validacion
from .calibracion import RegistroCalibracion
from .conocimiento import BaseConocimiento, Documento, RecuperadorTFIDF
from .elicitacion import SSR, Embedder, EmbedderLexico, ExtractorRating
from .enjambre import Enjambre, Reaccion, Resultado
from .llm import AnthropicClient, LLMClient, MockLLMClient
from .personas import GeneradorPersonas, Persona, Segmentacion

__version__ = "0.2.0"

__all__ = [
    "Enjambre",
    "Reaccion",
    "Resultado",
    "GeneradorPersonas",
    "Persona",
    "Segmentacion",
    "BaseConocimiento",
    "Documento",
    "RecuperadorTFIDF",
    "LLMClient",
    "MockLLMClient",
    "AnthropicClient",
    "ExtractorRating",
    "SSR",
    "Embedder",
    "EmbedderLexico",
    "RegistroCalibracion",
    "validacion",
]
