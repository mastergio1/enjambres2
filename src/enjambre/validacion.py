"""Validación: ¿la audiencia sintética reproduce la distribución objetivo?

Criterio de éxito de la Fase 1. Comparamos las marginales observadas en las
personas generadas contra las distribuciones objetivo y reportamos el error
(desviación máxima y distancia de variación total, TVD). Un error pequeño
significa que la audiencia no está sesgada hacia la "persona promedio".
"""
from __future__ import annotations

from collections import Counter

from .personas import Persona


def _normalizar(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values()) or 1.0
    return {k: v / total for k, v in dist.items()}


def error_dimension(personas: list[Persona], dimension: str, objetivo: dict[str, float]) -> dict:
    obj = _normalizar(objetivo)
    conteo = Counter(getattr(p, dimension) for p in personas)
    n = len(personas) or 1
    obs = {k: conteo.get(k, 0) / n for k in set(obj) | set(conteo)}
    claves = set(obj) | set(obs)
    desv_max = max(abs(obs.get(k, 0.0) - obj.get(k, 0.0)) for k in claves)
    tvd = 0.5 * sum(abs(obs.get(k, 0.0) - obj.get(k, 0.0)) for k in claves)
    return {"desv_max": desv_max, "tvd": tvd, "observado": obs, "objetivo": obj}


def error_distribucion(
    personas: list[Persona],
    objetivo: dict[str, dict[str, float]],
    *,
    excluir: tuple[str, ...] = (),
) -> dict[str, dict]:
    """Error por dimensión. ``excluir`` omite dimensiones muestreadas de forma
    condicional (su marginal difiere del objetivo por diseño)."""
    return {
        dim: error_dimension(personas, dim, dist)
        for dim, dist in objetivo.items()
        if dim not in excluir
    }


def reporte(
    personas: list[Persona],
    objetivo: dict[str, dict[str, float]],
    *,
    excluir: tuple[str, ...] = (),
    umbral: float = 0.05,
) -> str:
    res = error_distribucion(personas, objetivo, excluir=excluir)
    lineas = [f"  Validación de distribución (n={len(personas)}, umbral desv_max ≤ {umbral})"]
    lineas.append("  " + "-" * 46)
    ok_global = True
    for dim, m in res.items():
        ok = m["desv_max"] <= umbral
        ok_global = ok_global and ok
        marca = "✓" if ok else "✗"
        lineas.append(f"  {marca} {dim:<11} desv_max={m['desv_max']:.3f}  tvd={m['tvd']:.3f}")
    lineas.append("  " + "-" * 46)
    lineas.append(f"  {'✓ REPRODUCE el objetivo' if ok_global else '✗ revisar dimensiones marcadas'}")
    return "\n".join(lineas)
