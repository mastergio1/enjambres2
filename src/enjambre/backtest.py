"""Arnés de backtest de calibración retrospectiva (foco Chile).

Implementa el protocolo de los documentos de calibración:
- Persona chilena muestreada de distribuciones reales (Censo 2024 + GSE AIM).
- Cegado: el agente NUNCA ve la nota real ni el nº de reseñas del ítem.
- Elicitación por texto libre (método SSR); la nota se calcula después.
- Selección estratificada de ítems por rango de nota (fabrica dispersión).
- Scoring con doble métrica + línea base tonta + chequeos anti-autoengaño.

Advertencia GSE: los grupos C3, D y E no están verificados en fuente primaria.
Este arnés corre SOLO sobre los grupos confirmados (AB→C2) y lo declara. No se
inventan esos porcentajes (regla de la "advertencia bloqueante").
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

from .elicitacion import ANCLAS_CHILE, SSR
from .llm import LLMClient
from . import metricas


# --------------------------------------------------------------------------- #
# Persona chilena
# --------------------------------------------------------------------------- #
PROMPT_SISTEMA_CHILE = (
    "Eres un consumidor chileno real. No eres un asistente de IA, no rompes el personaje\n"
    "y no mencionas que eres una simulación. Piensas, hablas y decides como la persona\n"
    "descrita en tu PERFIL, con su nivel socioeconómico, su edad, su región y sus hábitos\n"
    "de consumo. Usas modismos y registro coherentes con ese perfil, sin exagerarlos.\n\n"
    "Tu tarea: reaccionar con honestidad a un ESTÍMULO (la descripción de un producto o\n"
    "aplicación) como lo harías en la vida real. Algunas cosas te van a gustar y otras no.\n"
    "No eres complaciente: si algo no te convence, lo dices; si te da lo mismo, lo dices.\n\n"
    "Reglas estrictas:\n"
    "- Responde SIEMPRE en primera persona y en español de Chile.\n"
    "- NO des una nota, puntaje, número de estrellas ni porcentaje. Nada de 'le pongo un 4'.\n"
    "- Expresa tu intención de compra/descarga con palabras, explicando por qué.\n"
    "- Basa tu reacción SOLO en lo que ves en el estímulo y en quién eres (tu perfil).\n"
    "  No inventes datos del producto que no estén en la descripción.\n"
    "- Sé concreto: menciona qué te atrae o qué te frena de este producto en particular."
)

ELICITACION_CHILE = (
    "Cuéntame, con tus palabras:\n"
    "1) ¿Qué es lo primero que piensas al ver esto?\n"
    "2) ¿Lo comprarías / descargarías? ¿Por qué sí o por qué no?\n"
    "3) ¿Qué tan probable es que lo recomiendes a alguien como tú?\n\n"
    "Responde natural, en 3-6 frases. No uses números ni notas."
)

# Descripción cualitativa por grupo GSE (para dar textura, no cifras).
_GSE_DESC = {
    "AB": "acomodado; compra online con frecuencia; menos sensible al precio; valora marca y experiencia.",
    "C1a": "acomodado; compra online con frecuencia; menos sensible al precio; valora marca y experiencia.",
    "C1b": "clase media; muy buscador de precio y ofertas; mezcla compra online y presencial.",
    "C2": "clase media; muy buscador de precio y ofertas; mezcla compra online y presencial.",
    "C3": "clase media baja; cuidadoso con el gasto; usa ferias y retail; compra online selectiva.",
    "D": "presupuesto ajustado; prioriza precio y utilidad; compra presencial predominante.",
    "E": "presupuesto ajustado; prioriza precio y utilidad; compra presencial predominante.",
}


def _grupo(gse: str) -> str:
    return {"AB": "alto", "C1a": "alto", "C1b": "medio", "C2": "medio",
            "C3": "medio_bajo", "D": "bajo", "E": "bajo"}.get(gse, "medio")


_DERIVADOS = {
    "alto": {"habito_digital": "alto", "sensibilidad_precio": "baja",
             "canal": "online frecuente", "educacion": "universitaria completa",
             "ocupacion": "profesional o gerencial"},
    "medio": {"habito_digital": "medio-alto", "sensibilidad_precio": "alta",
              "canal": "mixto online/presencial", "educacion": "técnica o media completa",
              "ocupacion": "empleado o técnico"},
    "medio_bajo": {"habito_digital": "medio", "sensibilidad_precio": "alta",
                   "canal": "retail y ferias, online selectiva", "educacion": "media completa",
                   "ocupacion": "cuenta propia o servicios"},
    "bajo": {"habito_digital": "bajo", "sensibilidad_precio": "muy alta",
             "canal": "presencial predominante", "educacion": "básica o media incompleta",
             "ocupacion": "trabajos ocasionales o informales"},
}

_TRAMO_BORDES = {"0-14": (0, 14), "15-64": (15, 64), "65+": (65, 85)}


@dataclass
class PersonaChile:
    id: str
    edad: int
    tramo_edad: str
    genero: str
    region: str
    gse: str
    gse_descripcion: str
    educacion: str
    ocupacion: str
    habito_digital: str
    sensibilidad_precio: str
    canal: str

    def perfil(self) -> str:
        return (
            "PERFIL:\n"
            f"- Edad: {self.edad} años (tramo {self.tramo_edad})\n"
            f"- Género: {self.genero}\n"
            f"- Región: {self.region}\n"
            f"- Nivel socioeconómico (GSE AIM): {self.gse}  → {self.gse_descripcion}\n"
            f"- Educación del hogar: {self.educacion}\n"
            f"- Ocupación: {self.ocupacion}\n"
            f"- Hábito digital: {self.habito_digital}\n"
            f"- Sensibilidad al precio: {self.sensibilidad_precio}\n"
            f"- Canal de compra habitual: {self.canal}"
        )

    def system_prompt(self, evidencia=None) -> str:  # evidencia: compat con Enjambre
        return f"{PROMPT_SISTEMA_CHILE}\n\n{self.perfil()}"

    def resumen(self) -> dict:
        return {
            "edad": self.edad, "tramo_edad": self.tramo_edad, "genero": self.genero,
            "region": self.region, "gse": self.gse, "educacion": self.educacion,
            "ocupacion": self.ocupacion, "sensibilidad_precio": self.sensibilidad_precio,
            "canal": self.canal,
        }


class GeneradorChile:
    """Muestrea personas chilenas de distribuciones reales (JSON)."""

    def __init__(self, datos: dict, semilla: int | None = 42, excluir_menores: bool = True) -> None:
        self.datos = datos
        self.excluir_menores = excluir_menores
        self._rng = random.Random(semilla)

    @classmethod
    def desde_json(cls, ruta: Path | str, **kw) -> "GeneradorChile":
        return cls(json.loads(Path(ruta).read_text(encoding="utf-8")), **kw)

    def _muestrear(self, dist: dict[str, float]) -> str:
        return self._rng.choices(list(dist), weights=list(dist.values()), k=1)[0]

    def generar(self, n: int) -> list[PersonaChile]:
        edades = {k: v for k, v in self.datos["edad_tramos"].items()
                  if not (self.excluir_menores and k == "0-14")}
        personas: list[PersonaChile] = []
        for i in range(n):
            genero = self._muestrear(self.datos["genero"])
            tramo = self._muestrear(edades)
            lo, hi = _TRAMO_BORDES.get(tramo, (18, 65))
            edad = self._rng.randint(lo, hi)
            region = self._muestrear(self.datos["region"])
            gse = self._muestrear(self.datos["gse"])
            der = _DERIVADOS[_grupo(gse)]
            personas.append(
                PersonaChile(
                    id=f"A{i + 1:04d}", edad=edad, tramo_edad=tramo, genero=genero,
                    region=region, gse=gse, gse_descripcion=_GSE_DESC.get(gse, ""),
                    educacion=der["educacion"], ocupacion=der["ocupacion"],
                    habito_digital=der["habito_digital"],
                    sensibilidad_precio=der["sensibilidad_precio"], canal=der["canal"],
                )
            )
        return personas


# --------------------------------------------------------------------------- #
# Ítems y selección estratificada
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    id: str
    tipo: str
    nombre: str
    categoria: str
    descripcion: str
    nota_real: float
    n_resenas: int

    def estimulo(self) -> str:
        # Cegado: sin nota, sin nº de reseñas, sin ranking.
        return (
            f"ESTÍMULO — Mira esta {self.tipo} y reacciona como lo harías tú:\n\n"
            f"Nombre: {self.nombre}\n"
            f"Categoría: {self.categoria}\n"
            f"Descripción:\n{self.descripcion}"
        )


def cargar_items(ruta: Path | str) -> list[Item]:
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return [Item(**d) for d in data]


def banda(nota: float) -> str:
    if nota <= 3.0:
        return "malo"
    if nota <= 3.9:
        return "medio"
    if nota <= 4.4:
        return "bueno"
    return "excelente"


PROPORCION_BANDAS = {"malo": 0.20, "medio": 0.30, "bueno": 0.30, "excelente": 0.20}


def seleccion_estratificada(
    items: list[Item], n: int = 50, min_resenas: int = 50, semilla: int | None = 0
) -> list[Item]:
    """Reparte la muestra por todo el rango de nota real para fabricar dispersión."""
    rng = random.Random(semilla)
    candidatos = [it for it in items if it.n_resenas >= min_resenas]
    por_banda: dict[str, list[Item]] = {b: [] for b in PROPORCION_BANDAS}
    for it in candidatos:
        por_banda[banda(it.nota_real)].append(it)
    seleccion: list[Item] = []
    for b, prop in PROPORCION_BANDAS.items():
        grupo = por_banda[b][:]
        rng.shuffle(grupo)
        seleccion.extend(grupo[: max(0, round(prop * n))])
    return seleccion


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class PrediccionItem:
    item: Item
    nota_predicha: float
    distribucion: list[float]  # media sobre [1,2,3,4,5]
    notas_agentes: list[float]
    casos: list[dict] = field(default_factory=list)


class Backtest:
    """Corre el arnés ciego sobre un conjunto de ítems y puntúa la calibración."""

    def __init__(
        self,
        llm: LLMClient,
        generador: GeneradorChile,
        ssr: SSR | None = None,
        n_agentes: int = 100,
        temperatura: float = 0.9,
    ) -> None:
        self.llm = llm
        self.generador = generador
        self.ssr = ssr or SSR(anclas=ANCLAS_CHILE)
        self.n_agentes = n_agentes
        self.temperatura = temperatura

    def predecir(self, item: Item, *, guardar_casos: bool = False) -> PrediccionItem:
        prompt = f"{item.estimulo()}\n\n{ELICITACION_CHILE}"
        acumulado = [0.0] * 5
        notas: list[float] = []
        casos: list[dict] = []
        for p in self.generador.generar(self.n_agentes):
            texto = self.llm.completar(p.system_prompt(), prompt, temperatura=self.temperatura)
            dist = self.ssr.distribucion(texto)
            if not dist:
                continue
            vec = [dist.get(n, 0.0) for n in range(1, 6)]
            nota = sum(n * v for n, v in zip(range(1, 6), vec))
            for i in range(5):
                acumulado[i] += vec[i]
            notas.append(nota)
            if guardar_casos:
                casos.append({
                    "agent_id": p.id, "item_id": item.id, "persona": p.resumen(),
                    "respuesta_texto": texto,
                    "ssr": {"distribucion_likert": vec, "nota_esperada": nota},
                })
        m = len(notas) or 1
        dist_media = [a / m for a in acumulado]
        nota_pred = sum(n * v for n, v in zip(range(1, 6), dist_media))
        return PrediccionItem(item, nota_pred, dist_media, notas, casos)

    def correr(self, items: list[Item], *, guardar_casos: bool = False) -> list[PrediccionItem]:
        return [self.predecir(it, guardar_casos=guardar_casos) for it in items]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _veredicto(r: float | None) -> str:
    if r is None:
        return "sin señal (varianza nula)"
    if r >= 0.85:
        return "EXCELENTE (nivel Nature/SSR)"
    if r >= 0.70:
        return "BUENO (screening)"
    if r >= 0.50:
        return "ACEPTABLE (solo exploratorio)"
    return "NO FIABLE"


def scoring(predicciones: list[PrediccionItem]) -> dict:
    reales = [p.item.nota_real for p in predicciones]
    pred = [p.nota_predicha for p in predicciones]
    r = metricas.pearson(pred, reales)
    disp_muestra = metricas.describe(reales)["std"]
    # homogeneidad: dispersión media de los agentes DENTRO de cada ítem
    homog = mean(pstdev(p.notas_agentes) for p in predicciones if len(p.notas_agentes) > 1) \
        if any(len(p.notas_agentes) > 1 for p in predicciones) else 0.0
    return {
        "n_items": len(predicciones),
        "correlacion_r": r,
        "veredicto": _veredicto(r),
        "similitud_forma": metricas.similitud_forma(pred, reales),
        "mae_modelo": metricas.mae(pred, reales),
        "mae_baseline_media": metricas.mae_baseline_media(reales),
        "dispersion_muestra_std": disp_muestra,
        "muestra_valida": disp_muestra >= 0.4,
        "homogeneidad_intra_item_std": homog,
        "agentes_variados": homog >= 0.3,
    }


def reporte(sc: dict) -> str:
    r = sc["correlacion_r"]
    r_txt = f"{r:.3f}" if r is not None else "n/d"
    gana = sc["mae_modelo"] < sc["mae_baseline_media"]
    L = [
        f"  Backtest de calibración (n={sc['n_items']} ítems)",
        "  " + "-" * 52,
        f"  Correlación r ...........: {r_txt}   → {sc['veredicto']}",
        f"  Similitud de forma (1-KS): {sc['similitud_forma']:.3f}   (objetivo ≥ 0.85)",
        f"  MAE modelo ..............: {sc['mae_modelo']:.3f}",
        f"  MAE baseline (media) ....: {sc['mae_baseline_media']:.3f}   "
        f"→ {'✓ modelo gana' if gana else '✗ no supera a la media'}",
        "  " + "-" * 52,
        "  Controles anti-autoengaño:",
        f"  · Dispersión de la muestra: std={sc['dispersion_muestra_std']:.3f}   "
        f"{'✓ válida' if sc['muestra_valida'] else '✗ muestra plana: el test no vale'}",
        f"  · Variedad de agentes .....: std={sc['homogeneidad_intra_item_std']:.3f}   "
        f"{'✓ opinan distinto' if sc['agentes_variados'] else '✗ homogéneos: sospechoso'}",
    ]
    return "\n".join(L)
