"""Demo de la Fase 2: método SSR + LLM real (opcional).

Uso:
    python examples/demo_fase2.py

- Muestra cómo SSR convierte una reacción en texto a intención de compra 1-5
  (con el embedder léxico offline; en prod se enchufa un embedder semántico).
- Corre un pre-test con SSR como elicitador (LLM simulado, sin llaves).
- Si defines ANTHROPIC_API_KEY, además reacciona con Claude real.
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import (  # noqa: E402
    SSR,
    BaseConocimiento,
    Enjambre,
    GeneradorPersonas,
    MockLLMClient,
    Segmentacion,
)

DATOS = RAIZ / "data"


def main() -> None:
    ssr = SSR()  # EmbedderLexico por defecto (offline)

    print("\n  ENJAMBRE 2 · Fase 2 — SSR (intención desde texto libre)\n")
    ejemplos = [
        "No me interesa para nada, jamás gastaría en esto.",
        "No sé, me da un poco igual, tal vez lo piense.",
        "¡Me encanta! Definitivamente lo compraría sin dudar.",
    ]
    for txt in ejemplos:
        dist = ssr.distribucion(txt)
        barras = " ".join(f"{n}★{dist[n]*100:>3.0f}%" for n in sorted(dist))
        print(f"  «{txt[:46]:<46}»  → {ssr.intencion(txt):.2f}/5")
        print(f"     {barras}")

    # Pre-test A/B con SSR como elicitador (LLM simulado)
    seg = Segmentacion.desde_json(DATOS / "segmentos_latam.example.json")
    base = BaseConocimiento.desde_json(DATOS / "resenas.example.json")
    enjambre = Enjambre(
        llm=MockLLMClient(),
        generador=GeneradorPersonas(seg, semilla=7),
        elicitador=ssr,
        conocimiento=base,
    )
    print("\n  " + "-" * 60)
    print("  Pre-test A/B con SSR (LLM simulado):\n")
    variantes = {
        "A · Salud": "Cuida tu cuerpo: 20g de proteína, cero azúcar.",
        "B · Precio": "El yogurt proteico más rico y accesible. Rinde y te llena.",
        "C · Sostenible": "Proteína real en botella retornable. Bueno para el planeta.",
    }
    estimulo = "Yogurt bebible de proteína sin azúcar en botella retornable"
    for puesto, (nombre, res) in enumerate(
        enjambre.comparar(variantes, n_personas=60, contexto=estimulo), 1
    ):
        print(f"  #{puesto}  {nombre:<16} intención {res.media():.2f}/5 · top-2-box {res.top_2_box():.0f}%")

    # LLM real, solo si hay llave
    print("\n  " + "-" * 60)
    if os.environ.get("ANTHROPIC_API_KEY"):
        from enjambre import AnthropicClient

        print("  Reacción con Claude real (una persona):\n")
        enj_claude = Enjambre(
            llm=AnthropicClient(modelo="claude-sonnet-5"),
            generador=GeneradorPersonas(seg, semilla=7),
            elicitador=ssr,
            conocimiento=base,
        )
        res = enj_claude.reaccionar(estimulo, n_personas=1, contexto="lanzamiento en México")
        r = res.reacciones[0]
        print(f"  Persona: {r.persona.descripcion()}")
        print(f"  Dijo: {r.texto.strip()}")
        print(f"  SSR → intención de compra: {r.intencion:.2f}/5")
    else:
        print("  (Define ANTHROPIC_API_KEY para ver el enjambre reaccionar con Claude real.)")
    print()


if __name__ == "__main__":
    main()
