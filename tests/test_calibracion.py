"""Tests del cierre de calibración: embedders, runner y detección de señal."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "examples"))

from enjambre import GeneradorChile, MockLLMClient, crear_embedder  # noqa: E402
from enjambre.backtest import Backtest, scoring  # noqa: E402
from enjambre.calibrar import correr_calibracion  # noqa: E402
from enjambre.dataset import cargar_dataset_ejemplo, seleccion_estratificada  # noqa: E402
from enjambre.elicitacion import EmbedderLexico  # noqa: E402

DATOS = RAIZ / "data"


def _gen():
    return GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=7)


def _dataset():
    return cargar_dataset_ejemplo(DATOS / "items_backtest.example.json")


def test_fabrica_embedder_lexico():
    emb = crear_embedder("lexico")
    assert isinstance(emb, EmbedderLexico)
    vecs = emb.embed(["hola mundo", "otra cosa"])
    assert len(vecs) == 2 and len(vecs[0]) == emb.dim
    # denso y L2-normalizado
    import math
    assert abs(math.sqrt(sum(x * x for x in vecs[0])) - 1.0) < 1e-9


def test_fabrica_embedder_desconocido():
    import pytest

    with pytest.raises(ValueError):
        crear_embedder("no-existe")


def test_runner_persiste_reporte_de_validacion(tmp_path):
    rep, preds, destino = correr_calibracion(
        _dataset(),
        nombre_embedder="lexico", llm=MockLLMClient(),
        n_agentes=10, n_items=8, semilla=0,
        salida_dir=tmp_path, registrar=False,
    )
    assert destino.exists()
    # dataset de ejemplo + mock ⇒ VALIDACION_ARNES, con advertencia y prefijo claro
    assert rep["tipo_corrida"] == "VALIDACION_ARNES"
    assert "_ADVERTENCIA" in rep
    assert destino.name.startswith("VALIDACION-ARNES_")
    assert rep["metadata"]["embedder"] == "lexico"
    assert rep["metadata"]["sha1_estimulos"] and rep["metadata"]["sha1_clave"]
    assert rep["metadata"]["version_anclas"] == "ANCLAS_CHILE/v1"
    assert len(rep["items"]) == len(preds)


def test_el_arnes_detecta_senal_con_lector_que_lee():
    """Control clave: mock ciego NO da señal; lector que comprende SÍ."""
    from validar_arnes import LLMHeuristico

    dataset = _dataset()
    sel = seleccion_estratificada(dataset, n=12, min_muestras=50, semilla=0)

    def r(llm):
        preds = Backtest(llm=llm, generador=_gen(), n_agentes=30).correr(sel)
        return scoring(preds, dataset)

    sc_ciego = r(MockLLMClient())
    sc_lee = r(LLMHeuristico())

    # el mock ciego no supera al baseline; el lector heurístico sí y correlaciona
    assert sc_ciego["mae_modelo"] >= sc_ciego["mae_baseline_media"] * 0.95
    assert sc_lee["correlacion_r"] is not None and sc_lee["correlacion_r"] > 0.4
    assert sc_lee["mae_modelo"] < sc_lee["mae_baseline_media"]
    assert sc_lee["correlacion_r"] > (sc_ciego["correlacion_r"] or -1)
