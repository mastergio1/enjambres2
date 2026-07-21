"""Demo: anclar personas en opiniones reales de X (RAG).

    python examples/demo_rag_x.py

Muestra qué opiniones reales recupera el enjambre para un estímulo, y cómo el
reporte de cobertura declara honestamente cuánto se ancló (o no) cada persona.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import Enjambre, GeneradorPersonas, MockLLMClient, cargar_opiniones_x  # noqa: E402
from enjambre.personas import Segmentacion  # noqa: E402

DATOS = RAIZ / "data"


def main() -> None:
    base = cargar_opiniones_x(DATOS / "opiniones_x_por_pais.json")  # excluye memes
    print(f"\n  Corpus RAG desde X: {len(base.documentos)} opiniones serias "
          f"({base.memes_excluidos} memes separados, no borrados)")
    for m in base.memes:
        print(f"   (meme fuera del corpus) [{m.tags.get('pais')}] {m.texto[:60]}...")
    print()

    estimulo = "Detergente líquido concentrado, rinde más y cuida los colores"
    print(f"  Estímulo: {estimulo}")
    print("  Opiniones recuperadas para una persona mexicana:")
    for d in base.recuperar(estimulo, k=3, tags_preferidos={"pais": "México"}):
        print(f"   · [{d.tags.get('pais')}/{d.tags.get('producto')}] {d.texto[:70]}...")

    seg = Segmentacion.desde_json(DATOS / "segmentos_latam.example.json")
    enjambre = Enjambre(
        llm=MockLLMClient(), generador=GeneradorPersonas(seg, semilla=7),
        conocimiento=base, k_contexto=2,
    )
    res = enjambre.reaccionar(estimulo, n_personas=40, contexto="detergente")
    cob = res.cobertura_rag
    print("\n  Cobertura RAG (honesta):")
    print(f"   con_corpus={cob['con_corpus']}  docs={cob['docs_en_corpus']}  "
          f"recuperados/persona={cob['recuperados_promedio']}  "
          f"sin_evidencia={cob['personas_sin_evidencia']}")
    if cob["advertencia"]:
        print(f"   ⚠ {cob['advertencia']}")
    else:
        print("   ✓ audiencia anclada en lenguaje real")
    print()


if __name__ == "__main__":
    main()
