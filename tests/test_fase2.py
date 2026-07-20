"""Tests de la Fase 2: método SSR y elicitación intercambiable."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import (  # noqa: E402
    SSR,
    Enjambre,
    ExtractorRating,
    GeneradorPersonas,
    MockLLMClient,
)


def test_ssr_rango_y_distribucion():
    ssr = SSR()
    dist = ssr.distribucion("Me encanta, definitivamente lo compraría.")
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    val = ssr.intencion("Me encanta, definitivamente lo compraría.")
    assert 1.0 <= val <= 5.0


def test_ssr_es_monotono():
    ssr = SSR()
    negativo = ssr.intencion("No me interesa para nada, jamás lo compraría.")
    positivo = ssr.intencion("Me encanta, definitivamente lo compraría.")
    assert positivo > negativo


def test_ssr_texto_vacio():
    ssr = SSR()
    assert ssr.intencion("") is None
    assert ssr.distribucion("   ") == {}


def test_elicitadores_dan_instruccion_distinta():
    assert "X/5" in ExtractorRating().instruccion_prompt()
    assert "número" in SSR().instruccion_prompt()


def test_enjambre_con_ssr_produce_resultado():
    enj = Enjambre(
        llm=MockLLMClient(),
        generador=GeneradorPersonas(semilla=3),
        elicitador=SSR(),
    )
    res = enj.reaccionar("Producto de prueba", n_personas=20)
    assert len(res.reacciones) == 20
    assert 1.0 <= res.media() <= 5.0
    assert 0.0 <= res.top_2_box() <= 100.0
    # la intención de SSR es continua
    assert any(not float(v).is_integer() for v in res.intenciones())


def test_extractor_sigue_funcionando():
    enj = Enjambre(llm=MockLLMClient(), generador=GeneradorPersonas(semilla=3))
    res = enj.reaccionar("Producto", n_personas=15)
    assert all(float(v).is_integer() for v in res.intenciones())
