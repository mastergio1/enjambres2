"""Tests de la Fase 1: personas ancladas, RAG y validación de distribución."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import (  # noqa: E402
    BaseConocimiento,
    Documento,
    Enjambre,
    GeneradorPersonas,
    MockLLMClient,
    Segmentacion,
    validacion,
)
from enjambre.conocimiento import RecuperadorTFIDF  # noqa: E402

DATOS = RAIZ / "data"


def _seg() -> Segmentacion:
    return Segmentacion.desde_json(DATOS / "segmentos_latam.example.json")


def test_carga_segmentacion_json():
    seg = _seg()
    assert "México" in seg.dimensiones["pais"]
    assert seg.condicionales["edad"]["segun"] == "pais"
    assert seg.intereses_por_nse["A/B (alto)"]


def test_reproduce_distribucion_objetivo():
    personas = GeneradorPersonas(_seg(), semilla=1).generar(4000)
    res = validacion.error_distribucion(personas, _seg().dimensiones, excluir=("edad",))
    # Con n grande, cada marginal muestreada debe reproducir el objetivo.
    assert all(m["desv_max"] <= 0.05 for m in res.values())


def test_condicional_edad_por_pais():
    personas = GeneradorPersonas(_seg(), semilla=2).generar(6000)
    def pct_joven(pais):
        sub = [p for p in personas if p.pais == pais]
        return sum(1 for p in sub if p.edad == "18-24") / (len(sub) or 1)
    # México está configurado más joven que Argentina.
    assert pct_joven("México") > pct_joven("Argentina")


def test_priors_ocean_desplazan_media():
    personas = GeneradorPersonas(_seg(), semilla=3).generar(3000)
    alto = [p for p in personas if p.nse == "A/B (alto)"]
    bajo = [p for p in personas if p.nse == "D/E (bajo)"]
    ap_alto = sum(p.ocean["apertura"] for p in alto) / len(alto)
    ap_bajo = sum(p.ocean["apertura"] for p in bajo) / len(bajo)
    assert ap_alto > ap_bajo  # A/B tiene prior de apertura +, D/E -


def test_rag_recupera_relevante():
    docs = [
        Documento("me encanta la botella retornable y lo sostenible", {"pais": "Chile"}),
        Documento("busco el precio más bajo, que sea barato", {"pais": "México"}),
    ]
    rec = RecuperadorTFIDF(docs)
    top = rec.recuperar("producto sostenible con empaque retornable", k=1)
    assert top and "retornable" in top[0].texto


def test_boost_por_tags():
    docs = [
        Documento("proteína y sabor", {"pais": "Argentina", "nse": "C (medio)"}),
        Documento("proteína y sabor", {"pais": "México", "nse": "D/E (bajo)"}),
    ]
    rec = RecuperadorTFIDF(docs)
    top = rec.recuperar("proteína sabor", k=1, tags_preferidos={"pais": "México"})
    assert top[0].tags["pais"] == "México"


def test_enjambre_con_conocimiento_inyecta_evidencia():
    base = BaseConocimiento.desde_json(DATOS / "resenas.example.json")
    enj = Enjambre(MockLLMClient(), GeneradorPersonas(_seg(), semilla=5), conocimiento=base)
    persona = GeneradorPersonas(_seg(), semilla=5).generar(1)[0]
    ev = enj._evidencia("yogurt de proteína sin azúcar", "", persona)
    assert isinstance(ev, list) and len(ev) >= 1
