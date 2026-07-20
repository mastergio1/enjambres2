"""Demo end-to-end del Enjambre 2 (corre sin API keys, con LLM simulado).

Uso:
    python examples/demo.py

Simula un pre-test A/B/C de tres variantes de mensaje para el mismo producto
y las rankea por intención de compra de una audiencia sintética LATAM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enjambre import Enjambre, GeneradorPersonas, MockLLMClient  # noqa: E402


def barra(pct: float, ancho: int = 24) -> str:
    llenos = round(pct / 100 * ancho)
    return "█" * llenos + "·" * (ancho - llenos)


def main() -> None:
    enjambre = Enjambre(llm=MockLLMClient(), generador=GeneradorPersonas(semilla=7))

    producto = "Yogurt bebible de proteína, sin azúcar, en botella retornable"
    variantes = {
        "A · Salud": "Cuida tu cuerpo: 20g de proteína, cero azúcar. Tu snack inteligente.",
        "B · Precio": "El yogurt proteico más rico y accesible. Rinde, cunde y te llena.",
        "C · Sostenible": "Proteína real en botella retornable. Bueno para ti, bueno para el planeta.",
    }

    print("\n  ENJAMBRE 2 · Pre-test de audiencias sintéticas (LATAM)")
    print("  Producto:", producto)
    print("  Audiencia: 40 consumidores sintéticos · LLM: simulado (demo)\n")
    print("  " + "-" * 60)

    ranking = enjambre.comparar(variantes, n_personas=40, contexto=producto)

    for puesto, (nombre, res) in enumerate(ranking, 1):
        print(f"\n  #{puesto}  {nombre}")
        print(f"       intención media : {res.media():.2f}/5   (±{res.dispersion():.2f})")
        print(f"       top-2-box (4-5) : {res.top_2_box():.0f}%   {barra(res.top_2_box())}")
        dist = res.distribucion()
        detalle = "  ".join(f"{k}★:{v}" for k, v in dist.items())
        print(f"       distribución    : {detalle}")

    ganadora = ranking[0][0]
    print("\n  " + "-" * 60)
    print(f"  ▶ Recomendación del enjambre: lanzar la variante «{ganadora}».")
    print("    (Demo con LLM simulado: los números son ilustrativos, no reales.)\n")


if __name__ == "__main__":
    main()
