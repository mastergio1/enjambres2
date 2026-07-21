"""Tests del cierre de calibración: embedders, runner y detección de señal."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "examples"))

from enjambre import GeneradorChile, MockLLMClient, crear_embedder  # noqa: E402
from enjambre.backtest import (  # noqa: E402
    Backtest,
    cargar_items,
    scoring,
    seleccion_estratificada,
)
from enjambre.calibrar import correr_calibracion  # noqa: E402
from enjambre.elicitacion import EmbedderLexico  # noqa: E402

DATOS = RAIZ / "data"


def _gen():
    return GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=7)


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


def test_runner_persiste_reporte(tmp_path):
    rep, preds, destino = correr_calibracion(
        DATOS / "items_backtest.example.json",
        nombre_embedder="lexico", llm=MockLLMClient(),
        n_agentes=10, n_items=8, semilla=0,
        salida_dir=tmp_path, registrar=False,
    )
    assert destino.exists()
    assert rep["metadata"]["embedder"] == "lexico"
    assert rep["metadata"]["dataset_sha1"]
    assert rep["metadata"]["version_anclas"] == "ANCLAS_CHILE/v1"
    assert len(rep["items"]) == len(preds)


def test_el_arnes_detecta_senal_con_lector_que_lee():
    """Control clave: mock ciego NO da señal; lector que comprende SÍ."""
    from demo_calibracion_real import LLMHeuristico

    items = cargar_items(DATOS / "items_backtest.example.json")
    sel = seleccion_estratificada(items, n=12, min_resenas=50, semilla=0)

    def r(llm):
        preds = Backtest(llm=llm, generador=_gen(), n_agentes=30).correr(sel)
        return scoring(preds)

    sc_ciego = r(MockLLMClient())
    sc_lee = r(LLMHeuristico())

    # el mock ciego no supera al baseline; el lector heurístico sí y correlaciona
    assert sc_ciego["mae_modelo"] >= sc_ciego["mae_baseline_media"] * 0.95
    assert sc_lee["correlacion_r"] is not None and sc_lee["correlacion_r"] > 0.4
    assert sc_lee["mae_modelo"] < sc_lee["mae_baseline_media"]
    assert sc_lee["correlacion_r"] > (sc_ciego["correlacion_r"] or -1)
