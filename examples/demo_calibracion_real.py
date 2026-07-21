"""Autoverificación de la calibración (offline, sin llaves).

El problema: con el LLM simulado (que no lee el estímulo) la correlación es ~0,
así que no prueba que el arnés funcione cuando el lector SÍ comprende. Aquí
usamos un LECTOR HEURÍSTICO que lee la descripción del ítem y reacciona según su
sentimiento —una versión pobre de lo que hace Claude—. Si el arnés surface señal
con este lector tonto-pero-que-lee, confiamos en que la cañería (persona ciega →
reacción → SSR → agregación → scoring) está bien y lista para Claude real.

    python examples/demo_calibracion_real.py

Esto NO es un resultado de calibración: es una prueba de que la máquina detecta
señal. El número real se mide con Claude + un embedder semántico sobre datos con
nota real (ver README / docs/calibracion.md).
"""
import hashlib
import sys
import unicodedata
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
from enjambre.llm import LLMClient  # noqa: E402

DATOS = RAIZ / "data"


def _norm(t: str) -> str:
    d = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in d if not unicodedata.combining(c))


class LLMHeuristico(LLMClient):
    """Lector de juguete que SÍ lee la descripción y reacciona por sentimiento.

    Stand-in offline de Claude para autoverificar el arnés. En producción se
    reemplaza por AnthropicClient, que lee mucho mejor.
    """

    POS = ["ahorr", "seguro", "facil", "gratis", "util", "recuerda", "comunidad",
           "fresc", "verificad", "simple", "sincro", "metas", "ciclovia",
           "domicilio", "efectivo", "sin datos", "recordatorio"]
    NEG = ["anuncio", "publicidad", "marca de agua", "cara", "permiso", "sin razon",
           "desactualiza", "limitad", "cobra", "suscripcion", "relampago", "notificacion"]
    FRASES = {
        1: "La verdad no me sirve, se ve puro problema y no lo descargaría.",
        2: "No me convence mucho, le veo varias cosas malas, probablemente no.",
        3: "Me da lo mismo, ni fu ni fa, quizás sí quizás no.",
        4: "Se ve bastante bueno y útil, probablemente lo descargaría.",
        5: "Me encanta, es justo lo que necesito, sin duda lo descargaría.",
    }

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:
        t = _norm(prompt)
        score = sum(t.count(w) for w in self.POS) - sum(t.count(w) for w in self.NEG)
        base = 3 + (score >= 1) + (score >= 2) - (score <= -1) - (score <= -2)
        jitter = (int(hashlib.sha256(sistema.encode()).hexdigest(), 16) % 3) - 1
        nivel = min(5, max(1, base + jitter))
        return self.FRASES[nivel]


def _correr(llm, etiqueta, sel, generador, n=40):
    preds = Backtest(llm=llm, generador=generador, n_agentes=n).correr(sel)
    print(f"\n  === Lector: {etiqueta} ===")
    print(reporte(scoring(preds)))
    return preds


def main() -> None:
    items = cargar_items(DATOS / "items_backtest.example.json")
    sel = seleccion_estratificada(items, n=12, min_resenas=50, semilla=0)
    generador = GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=7)

    print("\n  ENJAMBRE 2 · Autoverificación de calibración (offline)")
    print(f"  Ítems estratificados: {len(sel)}")

    # 1) Mock ciego: no lee → correlación ~0 (control negativo)
    _correr(MockLLMClient(), "mock ciego (control negativo)", sel, generador)

    # 2) Lector heurístico: lee la descripción → el arnés debe surface señal
    preds = _correr(LLMHeuristico(), "heurístico que SÍ lee (prueba de máquina)", sel, generador)

    print("\n  Predicho vs real (lector heurístico):")
    for p in sorted(preds, key=lambda x: x.item.nota_real, reverse=True):
        print(f"    {p.item.nombre:<18} real {p.item.nota_real:.1f}  ·  predicho {p.nota_predicha:.2f}")
    print("\n  ▶ Si el heurístico ya surface señal, con Claude real (que lee mejor)")
    print("    el arnés está listo para medir la calibración de verdad.\n")


if __name__ == "__main__":
    main()
