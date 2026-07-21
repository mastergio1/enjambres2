"""VALIDACIÓN DEL ARNÉS (no es calibración real).

Prueba que el instrumento de medición DETECTA SEÑAL cuando la hay. Con el LLM
simulado (que no lee el estímulo) la correlación es ~0; con un lector heurístico
que SÍ lee la descripción, el arnés surface señal (r>0). Eso confirma que la
cañería (persona ciega → reacción → SSR → agregación → scoring) está bien.

    python examples/validar_arnes.py

Esto NO mide el desempeño del producto. El número real se obtiene con Claude +
un embedder semántico sobre un dataset con notas reales (ver docs/calibracion.md).
"""
import hashlib
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import GeneradorChile, MockLLMClient  # noqa: E402
from enjambre.backtest import Backtest, reporte, scoring  # noqa: E402
from enjambre.dataset import cargar_dataset_ejemplo, seleccion_estratificada  # noqa: E402
from enjambre.llm import LLMClient  # noqa: E402

DATOS = RAIZ / "data"


def _norm(t: str) -> str:
    d = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in d if not unicodedata.combining(c))


class LLMHeuristico(LLMClient):
    """Lector de juguete que SÍ lee la descripción y reacciona por sentimiento.

    Stand-in offline de Claude para validar el arnés. es_real es False (heredado),
    así una corrida con este lector JAMÁS se deduce como calibración real.
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


def _correr(llm, etiqueta, estimulos, dataset, generador, n=40):
    preds = Backtest(llm=llm, generador=generador, n_agentes=n).correr(estimulos)
    print(f"\n  === Lector: {etiqueta} ===")
    print(reporte(scoring(preds, dataset)))
    return preds


def main() -> None:
    dataset = cargar_dataset_ejemplo(DATOS / "items_backtest.example.json")
    estimulos = seleccion_estratificada(dataset, n=12, min_muestras=50, semilla=0)
    generador = GeneradorChile.desde_json(DATOS / "segmentos_chile.example.json", semilla=7)

    print("\n  ENJAMBRE 2 · VALIDACIÓN DEL ARNÉS (no es calibración real)")
    print(f"  Estímulos ciegos: {len(estimulos)}")

    # 1) Mock ciego: no lee → correlación ~0 (control negativo)
    _correr(MockLLMClient(), "mock ciego (control negativo)", estimulos, dataset, generador)

    # 2) Lector heurístico: lee la descripción → el arnés debe surface señal
    preds = _correr(LLMHeuristico(), "heurístico que SÍ lee (prueba de máquina)",
                    estimulos, dataset, generador)

    print("\n  Predicho vs real (lector heurístico):")
    for p in sorted(preds, key=lambda x: dataset.resultado(x.estimulo.id), reverse=True):
        real = dataset.resultado(p.estimulo.id)
        print(f"    {p.estimulo.nombre:<18} real {real:.1f}  ·  predicho {p.nota_predicha:.2f}")
    print("\n  ▶ Si el heurístico ya surface señal, con Claude real (que lee mejor)")
    print("    el arnés está listo para medir la calibración de verdad.\n")


if __name__ == "__main__":
    main()
