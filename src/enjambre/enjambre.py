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
    intencion: int | None


@dataclass
class Resultado:
    estimulo: str
    reacciones: list[Reaccion]

    def intenciones(self) -> list[int]:
        return [r.intencion for r in self.reacciones if r.intencion is not None]

    def media(self) -> float:
        vals = self.intenciones()
        return mean(vals) if vals else 0.0

    def dispersion(self) -> float:
        vals = self.intenciones()
        return pstdev(vals) if len(vals) > 1 else 0.0

    def distribucion(self) -> dict[int, int]:
        dist = {i: 0 for i in range(1, 6)}
        for v in self.intenciones():
            dist[v] += 1
        return dist

    def top_2_box(self) -> float:
        """% de la audiencia con intención alta (4-5). Métrica estándar de la industria."""
        vals = self.intenciones()
        if not vals:
            return 0.0
        return 100 * sum(1 for v in vals if v >= 4) / len(vals)


class Enjambre:
    def __init__(
        self,
        llm: LLMClient,
        generador: GeneradorPersonas | None = None,
        extractor: ExtractorRating | None = None,
        conocimiento: BaseConocimiento | None = None,
        k_contexto: int = 3,
    ) -> None:
        self.llm = llm
        self.generador = generador or GeneradorPersonas()
        self.extractor = extractor or ExtractorRating()
        self.conocimiento = conocimiento
        self.k_contexto = k_contexto

    def _evidencia(self, estimulo: str, contexto: str, persona: Persona) -> list[str]:
        if not self.conocimiento:
            return []
        docs = self.conocimiento.recuperar(
            f"{estimulo} {contexto}".strip(),
            k=self.k_contexto,
            tags_preferidos={"pais": persona.pais, "nse": persona.nse},
        )
        return [d.texto for d in docs]

    def _prompt(self, estimulo: str, contexto: str) -> str:
        ctx = f"\nContexto: {contexto}" if contexto else ""
        return (
            f"Te muestran esto:{ctx}\n\n\"{estimulo}\"\n\n"
            "Reacciona en 1-2 frases con tu opinión sincera y luego indica tu "
            "intención de compra en formato 'Intención de compra: X/5' "
            "(1 = jamás, 5 = seguro lo compro)."
        )

    def reaccionar(
        self, estimulo: str, *, n_personas: int = 30, contexto: str = "", temperatura: float = 0.9
    ) -> Resultado:
        personas = self.generador.generar(n_personas)
        reacciones: list[Reaccion] = []
        prompt = self._prompt(estimulo, contexto)
        for p in personas:
            evidencia = self._evidencia(estimulo, contexto, p)
            texto = self.llm.completar(p.system_prompt(evidencia), prompt, temperatura=temperatura)
            reacciones.append(Reaccion(persona=p, texto=texto, intencion=self.extractor.extraer(texto)))
        return Resultado(estimulo=estimulo, reacciones=reacciones)

    def comparar(
        self, variantes: dict[str, str], *, n_personas: int = 30, contexto: str = ""
    ) -> list[tuple[str, Resultado]]:
        """Corre varias variantes y las devuelve rankeadas por intención media (desc)."""
        resultados = {
            nombre: self.reaccionar(est, n_personas=n_personas, contexto=contexto)
            for nombre, est in variantes.items()
        }
        return sorted(resultados.items(), key=lambda kv: kv[1].media(), reverse=True)
