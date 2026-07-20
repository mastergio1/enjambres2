"""Generador de personas sintéticas ancladas en segmentos LATAM.

El foso del producto es que las personas NO son inventadas al azar: se
muestrean de una distribución demográfica real (país, edad, nivel
socioeconómico, urbanidad, género) y se les asignan rasgos psicográficos
OCEAN. Cambiar la ``Segmentacion`` permite calibrar el enjambre para el
mercado específico de cada cliente.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


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

    def descripcion(self) -> str:
        rasgos_altos = [r for r, v in self.ocean.items() if v >= 4]
        matiz = f" Tiende a ser {', '.join(rasgos_altos)}." if rasgos_altos else ""
        return (
            f"Consumidor/a de {self.pais}, {self.edad} años, género {self.genero}, "
            f"nivel socioeconómico {self.nse}, entorno {self.urbanidad}.{matiz}"
        )

    def system_prompt(self) -> str:
        return (
            "Eres una persona real respondiendo de forma honesta y espontánea, "
            "con tu propio contexto cultural latinoamericano. No eres un asistente. "
            "Habla en primera persona, breve y natural.\n\n"
            f"Tu perfil: {self.descripcion()}"
        )


@dataclass
class Segmentacion:
    """Distribuciones de las que se muestrean las personas."""

    dimensiones: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in SEGMENTACION_LATAM_DEFAULT.items()}
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

    def _ocean(self) -> dict[str, int]:
        # Distribución centrada con algo de dispersión (escala 1-5).
        return {r: max(1, min(5, round(self._rng.gauss(3, 1)))) for r in RASGOS_OCEAN}

    def generar(self, n: int) -> list[Persona]:
        personas: list[Persona] = []
        dims = self.segmentacion.dimensiones
        for i in range(n):
            personas.append(
                Persona(
                    id=f"P{i + 1:03d}",
                    pais=self._muestrear(dims["pais"]),
                    edad=self._muestrear(dims["edad"]),
                    genero=self._muestrear(dims["genero"]),
                    nse=self._muestrear(dims["nse"]),
                    urbanidad=self._muestrear(dims["urbanidad"]),
                    ocean=self._ocean(),
                )
            )
        return personas
