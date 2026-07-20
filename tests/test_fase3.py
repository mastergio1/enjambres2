"""Tests de la Fase 3: arnés de backtest de calibración (Chile)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import GeneradorChile, MockLLMClient, SSR, metricas  # noqa: E402
from enjambre.backtest import (  # noqa: E402
    ANCLAS_CHILE,
    Backtest,
    Item,
    banda,
    cargar_items,
    scoring,
    seleccion_estratificada,
)

DATOS = RAIZ / "data"


def _gen() -> GeneradorChile:
    return GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=1)


def test_generador_chile_solo_grupos_verificados():
    personas = _gen().generar(500)
    assert {p.gse for p in personas} <= {"AB", "C1a", "C1b", "C2"}  # C3/D/E excluidos
    assert all(p.edad >= 15 for p in personas)  # menores excluidos
    assert all(p.region and p.canal for p in personas)


def test_persona_chile_prompt_no_pide_numero():
    p = _gen().generar(1)[0]
    sp = p.system_prompt()
    assert "consumidor chileno" in sp.lower()
    assert "PERFIL:" in sp
    assert "no des una nota" in sp.lower() or "NO des una nota" in sp


def test_item_estimulo_es_ciego():
    it = Item("x", "app", "Demo", "Cat", "una descripción", nota_real=4.6, n_resenas=900)
    est = it.estimulo()
    assert "Demo" in est and "descripción" in est
    assert "4.6" not in est and "900" not in est  # nota y nº reseñas ocultos


def test_banda_y_estratificacion():
    assert banda(2.5) == "malo" and banda(3.5) == "medio"
    assert banda(4.2) == "bueno" and banda(4.7) == "excelente"
    items = cargar_items(DATOS / "items_backtest.example.json")
    sel = seleccion_estratificada(items, n=12, min_resenas=50, semilla=0)
    bandas = {banda(it.nota_real) for it in sel}
    assert len(bandas) >= 3  # cubre varios tramos, no solo notas altas


def test_ssr_chile_monotono():
    ssr = SSR(anclas=ANCLAS_CHILE)
    neg = ssr.intencion("De ninguna manera descargaría esto, no me sirve para nada.")
    pos = ssr.intencion("Sin duda lo descargaría, me encanta y es justo lo que busco.")
    assert pos > neg


def test_ssr_promedio_de_varios_juegos():
    from enjambre.elicitacion import ANCLAS_DEFAULT
    ssr = SSR(anclas=[ANCLAS_CHILE, ANCLAS_DEFAULT])
    dist = ssr.distribucion("Me encanta, lo compraría sin dudarlo.")
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_backtest_corre_y_puntua():
    items = cargar_items(DATOS / "items_backtest.example.json")
    sel = seleccion_estratificada(items, n=8, min_resenas=50, semilla=0)
    bt = Backtest(llm=MockLLMClient(), generador=_gen(), n_agentes=15)
    preds = bt.correr(sel, guardar_casos=True)
    assert len(preds) == len(sel)
    assert all(1.0 <= p.nota_predicha <= 5.0 for p in preds)
    # el formato de salida por caso existe y no filtra la nota real
    caso = preds[0].casos[0]
    assert "respuesta_texto" in caso and "ssr" in caso
    sc = scoring(preds)
    assert "correlacion_r" in sc and "similitud_forma" in sc
    assert sc["dispersion_muestra_std"] > 0  # muestra estratificada tiene dispersión


def test_metricas_baseline_y_ks():
    reales = [2.0, 3.0, 4.0, 5.0]
    perfectas = [2.0, 3.0, 4.0, 5.0]
    assert metricas.pearson(perfectas, reales) > 0.99
    assert metricas.mae(perfectas, reales) == 0.0
    assert metricas.mae_baseline_media(reales) > 0
    assert metricas.similitud_forma(perfectas, reales) == 1.0
