"""Demo de la Fase 1: personas ancladas en datos reales.

Uso:
    python examples/demo_fase1.py

Muestra tres cosas:
  1. La audiencia sintética reproduce la distribución objetivo (validación).
  2. Las personas quedan aterrizadas con contexto RAG real (reseñas/tickets).
  3. El enjambre reacciona usando ese contexto.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import (  # noqa: E402
    BaseConocimiento,
    Enjambre,
    GeneradorPersonas,
    MockLLMClient,
    Segmentacion,
    validacion,
)

DATOS = RAIZ / "data"


def main() -> None:
    seg = Segmentacion.desde_json(DATOS / "segmentos_latam.example.json")
    base = BaseConocimiento.desde_json(DATOS / "resenas.example.json")
    generador = GeneradorPersonas(segmentacion=seg, semilla=7)

    print("\n  ENJAMBRE 2 · Fase 1 — Personas ancladas en datos reales\n")

    # 1) Validación: ¿reproduce la distribución objetivo?
    personas = generador.generar(2000)
    # 'edad' se muestrea condicionada al país, así que su marginal difiere por diseño.
    print(validacion.reporte(personas, seg.dimensiones, excluir=("edad",)))

    # Efecto de la condicional: México sale más joven que Argentina.
    def pct_joven(pais: str) -> float:
        sub = [p for p in personas if p.pais == pais]
        return 100 * sum(1 for p in sub if p.edad == "18-24") / (len(sub) or 1)

    print(f"\n  Condicional edad|país  →  18-24 en México: {pct_joven('México'):.0f}%"
          f"   vs Argentina: {pct_joven('Argentina'):.0f}%")

    # 2) Persona aterrizada con RAG
    estimulo = "Yogurt bebible de proteína, sin azúcar, en botella retornable"
    ejemplo = next(p for p in personas if p.nse == "D/E (bajo)" and p.pais == "México")
    evidencia = [d.texto for d in base.recuperar(
        estimulo, k=2, tags_preferidos={"pais": ejemplo.pais, "nse": ejemplo.nse})]
    print("\n  " + "-" * 60)
    print("  Ejemplo de persona aterrizada (system prompt que ve el LLM):\n")
    print("  " + ejemplo.system_prompt(evidencia).replace("\n", "\n  "))

    # 3) Reacción del enjambre CON contexto RAG
    enjambre = Enjambre(
        llm=MockLLMClient(), generador=generador, conocimiento=base, k_contexto=2
    )
    print("\n  " + "-" * 60)
    print("  Pre-test A/B con audiencia aterrizada (LLM simulado):\n")
    variantes = {
        "A · Salud": "Cuida tu cuerpo: 20g de proteína, cero azúcar.",
        "B · Precio": "El yogurt proteico más rico y accesible. Rinde y te llena.",
        "C · Sostenible": "Proteína real en botella retornable. Bueno para el planeta.",
    }
    ranking = enjambre.comparar(variantes, n_personas=60, contexto=estimulo)
    for puesto, (nombre, res) in enumerate(ranking, 1):
        print(f"  #{puesto}  {nombre:<16} intención {res.media():.2f}/5  · top-2-box {res.top_2_box():.0f}%")
    print(f"\n  ▶ Con LLM real (Fase 2) estas reacciones citarán el contexto RAG.\n")


if __name__ == "__main__":
    main()
