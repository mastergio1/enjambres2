"""Tests de humo del Enjambre 2 (corren sin API keys, con MockLLMClient)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enjambre import (  # noqa: E402
    Enjambre,
    ExtractorRating,
    GeneradorPersonas,
    MockLLMClient,
    RegistroCalibracion,
)


def test_generador_reproducible():
    a = GeneradorPersonas(semilla=1).generar(10)
    b = GeneradorPersonas(semilla=1).generar(10)
    assert [p.pais for p in a] == [p.pais for p in b]
    assert all(1 <= v <= 5 for p in a for v in p.ocean.values())


def test_extractor_rating():
    ex = ExtractorRating()
    assert ex.extraer("Reacción: me gusta. Intención de compra: 4/5.") == 4
    assert ex.extraer("me da 2/5") == 2
    assert ex.extraer("sin número") is None


def test_reaccionar_produce_distribucion():
    enj = Enjambre(llm=MockLLMClient(), generador=GeneradorPersonas(semilla=3))
    res = enj.reaccionar("Producto de prueba", n_personas=25)
    assert len(res.reacciones) == 25
    assert 1.0 <= res.media() <= 5.0
    assert 0.0 <= res.top_2_box() <= 100.0
    assert sum(res.distribucion().values()) == len(res.intenciones())


def test_comparar_rankea():
    enj = Enjambre(llm=MockLLMClient(), generador=GeneradorPersonas(semilla=5))
    ranking = enj.comparar({"X": "mensaje uno", "Y": "mensaje dos"}, n_personas=20)
    assert [n for n, _ in ranking] and ranking[0][1].media() >= ranking[-1][1].media()


def test_calibracion_correlacion_solo_real(tmp_path):
    from enjambre import TipoCorrida

    reg = RegistroCalibracion(tmp_path / "cal.db")
    # registros reales (buenos) + ruido de validación que NO debe contaminar
    for pred, real in [(2.0, 2.2), (3.0, 3.1), (4.0, 3.9), (5.0, 4.8)]:
        i = reg.registrar_prediccion("caso", "est", pred, tipo=TipoCorrida.CALIBRACION_REAL)
        reg.registrar_real(i, real)
    for pred, real in [(2.0, 5.0), (5.0, 1.0)]:  # validación con correlación invertida
        i = reg.registrar_prediccion("caso", "est", pred, tipo=TipoCorrida.VALIDACION_ARNES)
        reg.registrar_real(i, real)
    corr = reg.correlacion_real()  # debe ignorar la validación
    assert reg.contar(TipoCorrida.CALIBRACION_REAL) == 4
    reg.cerrar()
    assert corr is not None and corr > 0.9

    # tipo obligatorio: no se puede registrar sin él
    import pytest
    reg2 = RegistroCalibracion(tmp_path / "cal2.db")
    with pytest.raises(TypeError):
        reg2.registrar_prediccion("caso", "est", 3.0)
    reg2.cerrar()
