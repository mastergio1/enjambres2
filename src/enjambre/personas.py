"""Generador de personas sintéticas ancladas en segmentos LATAM.

El foso del producto es que las personas NO son inventadas al azar: se
muestrean de una distribución demográfica real (país, edad, nivel
socioeconómico, urbanidad, género), con rasgos psicográficos OCEAN sesgados
por segmento, e intereses por nivel socioeconómico.

Fase 1 añade:
- Carga de la segmentación desde datos del cliente (``Segmentacion.desde_json``).
- Distribuciones **condicionales** (p. ej. la edad varía según el país).
- **Priors OCEAN** por segmento (no todos los rasgos son uniformes).
- Intereses por segmento, y contexto RAG opcional en el ``system_prompt``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


# Distribución de referencia (aproximada, editable por cliente/campaña).
# Los pesos no necesitan sumar 1; se normalizan al muestrear.
SEGMENTACION_LATAM_DEFAULT: dict[str, dict[str, float]] = {
    "pais": {
        "México": 34, "Colombia": 12, "Argentina": 10, "Perú": 8,
        "Chile": 6, "Ecuador": 5, "Guatemala": 4, "Venezuela": 6,
        "Bolivia": 3, "Rep. Dominicana": 3, "Otros LATAM": 9,
    },
    "edad": {"18-24": 22, "25-34": 30, "35-44": 24, "45-54": 14, "55+": 10},
    "genero": {"femenino": 50, "masculino": 48, "no binario": 2},
    "nse": {"A/B (alto)": 12, "C (medio)": 45, "D/E (bajo)": 43},
    "urbanidad": {"urbano": 68, "periurbano": 20, "rural": 12},
}

RASGOS_OCEAN = ("apertura", "responsabilidad", "extraversion", "amabilidad", "neuroticismo")


@dataclass
class Persona:
    """Una persona sintética con perfil demográfico y psicográfico."""

    id: str
    pais: str
    edad: str
    genero: str
    nse: str
    urbanidad: str
    ocean: dict[str, int]  # cada rasgo en escala 1-5
    intereses: list[str] = field(default_factory=list)

    def descripcion(self) -> str:
        rasgos_altos = [r for r, v in self.ocean.items() if v >= 4]
        matiz = f" Tiendes a ser {', '.join(rasgos_altos)}." if rasgos_altos else ""
        gustos = f" Te interesan: {', '.join(self.intereses)}." if self.intereses else ""
        return (
            f"Consumidor/a de {self.pais}, {self.edad} años, género {self.genero}, "
            f"nivel socioeconómico {self.nse}, entorno {self.urbanidad}.{matiz}{gustos}"
        )

    def system_prompt(self, evidencia: list[str] | None = None) -> str:
        base = (
            "Eres una persona real respondiendo de forma honesta y espontánea, "
            "con tu propio contexto cultural latinoamericano. No eres un asistente. "
            "Habla en primera persona, breve y natural.\n\n"
            f"Tu perfil: {self.descripcion()}"
        )
        if evidencia:
            frag = "\n".join(f"- {e}" for e in evidencia)
            base += (
                "\n\nLo que gente parecida a ti ha dicho antes sobre productos "
                "similares (tenlo en cuenta; puedes coincidir o no):\n" + frag
            )
        return base


@dataclass
class Segmentacion:
    """Distribuciones y priors de los que se muestrean las personas.

    - ``dimensiones``: marginales por dimensión.
    - ``condicionales``: dependencias, p. ej. edad según país::

        {"edad": {"segun": "pais", "tablas": {"México": {"18-24": 0.3, ...}}}}

    - ``ocean_priors``: desplazamientos de la media OCEAN por segmento::

        {"nse": {"A/B (alto)": {"apertura": 0.5}}}

    - ``intereses_por_nse``: pool de intereses por nivel socioeconómico.
    """

    dimensiones: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in SEGMENTACION_LATAM_DEFAULT.items()}
    )
    condicionales: dict = field(default_factory=dict)
    ocean_priors: dict = field(default_factory=dict)
    intereses_por_nse: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def desde_json(cls, ruta: Path | str) -> "Segmentacion":
        data = json.loads(Path(ruta).read_text(encoding="utf-8"))
        base = {k: dict(v) for k, v in SEGMENTACION_LATAM_DEFAULT.items()}
        base.update(data.get("dimensiones", {}))
        return cls(
            dimensiones=base,
            condicionales=data.get("condicionales", {}),
            ocean_priors=data.get("ocean_priors", {}),
            intereses_por_nse=data.get("intereses_por_nse", {}),
        )


class GeneradorPersonas:
    """Muestrea personas de una segmentación de forma reproducible."""

    def __init__(self, segmentacion: Segmentacion | None = None, semilla: int | None = 42) -> None:
        self.segmentacion = segmentacion or Segmentacion()
        self._rng = random.Random(semilla)

    def _muestrear(self, distribucion: dict[str, float]) -> str:
        valores = list(distribucion.keys())
        pesos = list(distribucion.values())
        return self._rng.choices(valores, weights=pesos, k=1)[0]

    def _condicional(self, dimension: str, valores: dict[str, str]) -> str | None:
        cfg = self.segmentacion.condicionales.get(dimension)
        if not cfg:
            return None
        tabla = cfg.get("tablas", {}).get(valores.get(cfg.get("segun", "")))
        return self._muestrear(tabla) if tabla else None

    def _ocean(self, pais: str, nse: str) -> dict[str, int]:
        medias = {r: 3.0 for r in RASGOS_OCEAN}
        for ambito, clave in (("pais", pais), ("nse", nse)):
            priors = self.segmentacion.ocean_priors.get(ambito, {}).get(clave, {})
            for rasgo, shift in priors.items():
                if rasgo in medias:
                    medias[rasgo] += shift
        return {r: max(1, min(5, round(self._rng.gauss(medias[r], 1)))) for r in RASGOS_OCEAN}

    def _intereses(self, nse: str) -> list[str]:
        pool = self.segmentacion.intereses_por_nse.get(nse, [])
        if not pool:
            return []
        return self._rng.sample(pool, k=min(3, len(pool)))

    def generar(self, n: int) -> list[Persona]:
        personas: list[Persona] = []
        dims = self.segmentacion.dimensiones
        for i in range(n):
            pais = self._muestrear(dims["pais"])
            genero = self._muestrear(dims["genero"])
            nse = self._muestrear(dims["nse"])
            urbanidad = self._muestrear(dims["urbanidad"])
            edad = self._condicional("edad", {"pais": pais}) or self._muestrear(dims["edad"])
            personas.append(
                Persona(
                    id=f"P{i + 1:03d}",
                    pais=pais,
                    edad=edad,
                    genero=genero,
                    nse=nse,
                    urbanidad=urbanidad,
                    ocean=self._ocean(pais, nse),
                    intereses=self._intereses(nse),
                )
            )
        return personas
