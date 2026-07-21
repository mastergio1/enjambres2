"""Tests de robustez/costo portados del Enjambre 1: temperature, fallback, caché."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import LLMCache, MockLLMClient  # noqa: E402
from enjambre.llm import _acepta_temperature, reaccion_fallback  # noqa: E402


def test_sonnet5_no_acepta_temperature():
    # Sonnet 5 / Opus / Fable rechazan temperature (HTTP 400); haiku la acepta.
    assert _acepta_temperature("claude-sonnet-5") is False
    assert _acepta_temperature("claude-opus-4-8") is False
    assert _acepta_temperature("claude-fable-5") is False
    assert _acepta_temperature("claude-haiku-4-5-20251001") is True


def test_reaccion_fallback_es_determinista_y_neutra():
    a = reaccion_fallback("cualquier prompt")
    b = reaccion_fallback("otro prompt distinto")
    assert a == b and a.strip()  # neutra y determinista
    # el SSR la puntúa cerca del centro (no sesga)
    from enjambre import SSR
    assert 2.0 <= SSR().intencion(a) <= 4.0


class _MockContador(MockLLMClient):
    def __init__(self):
        self.llamadas = 0

    def completar(self, sistema, prompt, *, temperatura=0.9):
        self.llamadas += 1
        return super().completar(sistema, prompt, temperatura=temperatura)


def test_cache_evita_llamadas_repetidas():
    base = _MockContador()
    cache = LLMCache(base)
    cache.completar("sys", "hola")
    cache.completar("sys", "hola")  # mismo → cacheado
    cache.completar("sys", "otra")  # distinto → nueva llamada
    assert base.llamadas == 2  # no 3
    assert cache.hits == 1 and cache.misses == 2
    # espeja es_real para no romper la deducción del tipo de corrida
    assert cache.es_real == base.es_real


def test_cache_espeja_es_real():
    class _FalsoReal(MockLLMClient):
        es_real = True

    assert LLMCache(_FalsoReal()).es_real is True
    assert LLMCache(MockLLMClient()).es_real is False
