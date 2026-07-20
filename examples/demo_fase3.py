"""Demo de la Fase 3: backtest de calibración retrospectiva (foco Chile).

Uso:
    python examples/demo_fase3.py                 # LLM simulado (offline)
    ANTHROPIC_API_KEY=... python examples/demo_fase3.py   # con Claude real

IMPORTANTE: con el LLM simulado la correlación será ~0 A PROPÓSITO — el mock no
"lee" el estímulo, así que no puede predecir la nota real. Eso demuestra que el
arnés no hace trampa: la señal solo aparece con un LLM real que sí razona. Con
Claude real, este mismo arnés mide la calibración de verdad.
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import GeneradorChile, MockLLMClient  # noqa: E402
from enjambre.backtest import (  # noqa: E402
    Backtest,
    cargar_items,
    reporte,
    scoring,
    seleccion_estratificada,
)

DATOS = RAIZ / "data"


def main() -> None:
    items = cargar_items(DATOS / "items_backtest.example.json")
    seleccion = seleccion_estratificada(items, n=12, min_resenas=50, semilla=0)
    generador = GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=7)

    if os.environ.get("ANTHROPIC_API_KEY"):
        from enjambre import AnthropicClient

        llm = AnthropicClient(modelo="claude-sonnet-5")
        n_agentes = 20
        nota_llm = "Claude real"
    else:
        llm = MockLLMClient()
        n_agentes = 40
        nota_llm = "simulado (correlación ~0 esperada)"

    print("\n  ENJAMBRE 2 · Fase 3 — Backtest de calibración (Chile)")
    print(f"  Ítems: {len(seleccion)} (estratificados) · Agentes/ítem: {n_agentes} · LLM: {nota_llm}")
    print("  GSE: solo AB→C2 (C3/D/E sin verificar, excluidos a propósito)\n")

    bt = Backtest(llm=llm, generador=generador, n_agentes=n_agentes)
    predicciones = bt.correr(seleccion)

    print("  Predicho vs real (ciego):")
    for p in sorted(predicciones, key=lambda x: x.item.nota_real, reverse=True):
        print(f"    {p.item.nombre:<18} real {p.item.nota_real:.1f}  ·  predicho {p.nota_predicha:.2f}")

    print()
    print(reporte(scoring(predicciones)))
    print()


if __name__ == "__main__":
    main()
