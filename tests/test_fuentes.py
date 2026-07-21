"""Tests de la ingesta de opiniones de X al corpus RAG."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import cargar_opiniones_x  # noqa: E402
from enjambre.fuentes import _norm_pais, opiniones_x_a_documentos  # noqa: E402

DATOS = RAIZ / "data"


def test_normaliza_pais_a_valor_de_personas():
    assert _norm_pais("Mexico") == "México"  # sin acento → con acento
    assert _norm_pais("chile") == "Chile"


def test_aplana_estructura_anidada():
    datos = [{
        "pais": "Mexico",
        "productos": {
            "detergente": [{"text": "buen detergente", "sentimiento": "positivo", "engagement": 10}],
            "cafe": [{"text": "café justo de Chiapas"}],
        },
    }]
    docs = opiniones_x_a_documentos(datos)
    assert len(docs) == 2
    d = docs[0]
    assert d.tags["pais"] == "México" and d.tags["producto"] == "detergente"
    assert d.tags["sentimiento"] == "positivo" and d.fuente == "X"
    assert d.tags["engagement"] == "10"  # trazabilidad, string


def test_ignora_textos_vacios():
    docs = opiniones_x_a_documentos([{"pais": "Chile", "productos": {"leche": [{"text": "  "}]}}])
    assert docs == []


def test_carga_archivo_real_y_recupera_por_pais():
    base = cargar_opiniones_x(DATOS / "opiniones_x_por_pais.json")
    assert len(base.documentos) >= 8
    # una consulta de detergente en México debe traer opiniones mexicanas de detergente
    top = base.recuperar("detergente barato", k=2, tags_preferidos={"pais": "México"})
    assert top and any(d.tags["pais"] == "México" for d in top)
