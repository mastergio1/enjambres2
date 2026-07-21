"""Enjambre 2 — Motor de audiencias sintéticas para pre-test de productos (LATAM).

Producto núcleo de Rubicon Lab: subes un anuncio, concepto o mensaje y un
enjambre de consumidores sintéticos calibrados para mercados latinos
reacciona, rankea variantes y estima la intención de compra antes de lanzar.
"""
from . import backtest, calibrar, corrida, dataset, embeddings, metricas, validacion
from .backtest import Backtest, GeneradorChile, PersonaChile, scoring
from .calibracion import RegistroCalibracion
from .calibrar import correr_calibracion
from .conocimiento import BaseConocimiento, Documento, RecuperadorTFIDF
from .corrida import TipoCorrida, deducir_tipo
from .dataset import (
    DatasetCalibracion,
    EstimuloCiego,
    cargar_dataset_ciego,
    cargar_dataset_ejemplo,
    seleccion_estratificada,
)
from .elicitacion import ANCLAS_CHILE, SSR, Embedder, EmbedderLexico, ExtractorRating
from .embeddings import crear_embedder
from .fuentes import cargar_opiniones_x
from .enjambre import Enjambre, Reaccion, Resultado
from .llm import AnthropicClient, LLMCache, LLMClient, MockLLMClient
from .personas import GeneradorPersonas, Persona, Segmentacion

__version__ = "0.4.0"

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
    "cargar_opiniones_x",
    "LLMClient",
    "MockLLMClient",
    "AnthropicClient",
    "LLMCache",
    "ExtractorRating",
    "SSR",
    "Embedder",
    "EmbedderLexico",
    "ANCLAS_CHILE",
    "RegistroCalibracion",
    "Backtest",
    "scoring",
    "GeneradorChile",
    "PersonaChile",
    "TipoCorrida",
    "deducir_tipo",
    "DatasetCalibracion",
    "EstimuloCiego",
    "cargar_dataset_ciego",
    "cargar_dataset_ejemplo",
    "seleccion_estratificada",
    "crear_embedder",
    "correr_calibracion",
    "validacion",
    "metricas",
    "backtest",
    "corrida",
    "dataset",
    "embeddings",
    "calibrar",
]
