"""Orquestador del enjambre: estímulo -> reacción colectiva.

Toma un estímulo (un anuncio, un concepto de producto, un mensaje), genera
una audiencia sintética y recoge la reacción de cada persona. Agrega los
resultados en una distribución de intención de compra. Comparar varios
estímulos permite rankear variantes A/B antes de gastar en medios.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .conocimiento import BaseConocimiento
from .elicitacion import ExtractorRating
from .llm import LLMClient
from .personas import GeneradorPersonas, Persona


@dataclass
class Reaccion:
    persona: Persona
    texto: str
    intencion: float | None


@dataclass
class Resultado:
    estimulo: str
    reacciones: list[Reaccion]
    cobertura_rag: dict | None = None  # se rellena en Enjambre.reaccionar

    def intenciones(self) -> list[float]:
        return [r.intencion for r in self.reacciones if r.intencion is not None]

    def media(self) -> float:
        vals = self.intenciones()
        return mean(vals) if vals else 0.0

    def dispersion(self) -> float:
        vals = self.intenciones()
        return pstdev(vals) if len(vals) > 1 else 0.0

    def distribucion(self) -> dict[int, int]:
        """Cuenta por punto Likert (redondea la intención, que puede ser continua)."""
        dist = {i: 0 for i in range(1, 6)}
        for v in self.intenciones():
            dist[min(5, max(1, round(v)))] += 1
        return dist

    def top_2_box(self) -> float:
        """% de la audiencia con intención alta (≈4-5). Métrica estándar de la industria."""
        vals = self.intenciones()
        if not vals:
            return 0.0
        return 100 * sum(1 for v in vals if v >= 3.5) / len(vals)


class Enjambre:
    def __init__(
        self,
        llm: LLMClient,
        generador: GeneradorPersonas | None = None,
        elicitador=None,
        conocimiento: BaseConocimiento | None = None,
        k_contexto: int = 3,
    ) -> None:
        self.llm = llm
        self.generador = generador or GeneradorPersonas()
        # elicitador: cualquier objeto con .intencion(texto) e .instruccion_prompt()
        # (ExtractorRating por defecto; SSR para el método semántico).
        self.elicitador = elicitador or ExtractorRating()
        self.conocimiento = conocimiento
        self.k_contexto = k_contexto

    def _evidencia(self, estimulo: str, contexto: str, persona: Persona) -> list[str]:
        if not self.conocimiento:
            return []
        tags = {}
        pais = getattr(persona, "pais", None)
        nse = getattr(persona, "nse", None)
        if pais:
            tags["pais"] = pais
        if nse:
            tags["nse"] = nse
        docs = self.conocimiento.recuperar(
            f"{estimulo} {contexto}".strip(),
            k=self.k_contexto,
            tags_preferidos=tags or None,
        )
        return [d.texto for d in docs]

    def _cobertura(self, personas, evidencias: list[list[str]]) -> dict:
        """Reporta cuánto se ancló cada persona en lenguaje real (nunca en silencio)."""
        docs_corpus = len(self.conocimiento.documentos) if self.conocimiento else 0
        n = len(personas) or 1
        por_segmento: dict[str, dict] = {}
        total = 0
        sin_evidencia = 0
        for p, ev in zip(personas, evidencias):
            total += len(ev)
            if not ev:
                sin_evidencia += 1
            seg = f"{getattr(p, 'pais', None) or getattr(p, 'region', '?')}/" \
                  f"{getattr(p, 'nse', None) or getattr(p, 'gse', '?')}"
            slot = por_segmento.setdefault(seg, {"consultas": 0, "recuperados": 0})
            slot["consultas"] += 1
            slot["recuperados"] += len(ev)
        con_corpus = docs_corpus > 0
        cob = {
            "con_corpus": con_corpus,
            "docs_en_corpus": docs_corpus,
            "recuperados_promedio": round(total / n, 2),
            "personas_sin_evidencia": sin_evidencia,
            "por_segmento": por_segmento,
            "advertencia": None,
        }
        if not con_corpus:
            cob["advertencia"] = (
                "Corpus RAG vacío: las personas están ancladas DEMOGRÁFICAMENTE "
                "pero NO en lenguaje real (reseñas/tickets). Las reacciones salen "
                "del pre-entrenamiento del modelo, no de datos del cliente."
            )
        elif total / n < 0.5:
            cob["advertencia"] = (
                f"Cobertura RAG muy baja ({total / n:.2f} docs/persona): la mayoría "
                "de las personas no encontró lenguaje real relevante para este estímulo."
            )
        return cob

    def _prompt(self, estimulo: str, contexto: str) -> str:
        ctx = f"\nContexto: {contexto}" if contexto else ""
        return (
            f"Te muestran esto:{ctx}\n\n\"{estimulo}\"\n\n"
            + self.elicitador.instruccion_prompt()
        )

    def reaccionar(
        self, estimulo: str, *, n_personas: int = 30, contexto: str = "", temperatura: float = 0.9
    ) -> Resultado:
        personas = self.generador.generar(n_personas)
        reacciones: list[Reaccion] = []
        evidencias: list[list[str]] = []
        prompt = self._prompt(estimulo, contexto)
        for p in personas:
            evidencia = self._evidencia(estimulo, contexto, p)
            evidencias.append(evidencia)
            texto = self.llm.completar(p.system_prompt(evidencia), prompt, temperatura=temperatura)
            reacciones.append(
                Reaccion(persona=p, texto=texto, intencion=self.elicitador.intencion(texto))
            )
        return Resultado(
            estimulo=estimulo,
            reacciones=reacciones,
            cobertura_rag=self._cobertura(personas, evidencias),
        )

    def comparar(
        self, variantes: dict[str, str], *, n_personas: int = 30, contexto: str = ""
    ) -> list[tuple[str, Resultado]]:
        """Corre varias variantes y las devuelve rankeadas por intención media (desc)."""
        resultados = {
            nombre: self.reaccionar(est, n_personas=n_personas, contexto=contexto)
            for nombre, est in variantes.items()
        }
        return sorted(resultados.items(), key=lambda kv: kv[1].media(), reverse=True)
