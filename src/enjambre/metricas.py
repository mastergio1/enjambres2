"""Métricas para el scoring del backtest.

Doble métrica emparejada (como exige el paquete de calibración):
- correlación de las notas (¿predice el orden y la magnitud?), y
- forma de la distribución (¿la nube de predicciones se parece a la real, o
  está aplastada hacia el centro?).

Más una línea base tonta ("predecir siempre la media") contra la que el modelo
debe ganar claramente, y utilidades de dispersión.
"""
from __future__ import annotations

from statistics import mean, pstdev


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx**0.5 * vy**0.5)


def mae(xs: list[float], ys: list[float]) -> float:
    return mean(abs(x - y) for x, y in zip(xs, ys)) if xs else 0.0


def mae_baseline_media(reales: list[float]) -> float:
    """Error de la línea base tonta: predecir siempre la media real."""
    if not reales:
        return 0.0
    m = mean(reales)
    return mean(abs(r - m) for r in reales)


def ks_dos_muestras(a: list[float], b: list[float]) -> float:
    """Estadístico KS de dos muestras (0 = idénticas, 1 = disjuntas)."""
    if not a or not b:
        return 1.0
    na, nb = len(a), len(b)
    d = 0.0
    for x in sorted(set(a) | set(b)):
        fa = sum(1 for v in a if v <= x) / na
        fb = sum(1 for v in b if v <= x) / nb
        d = max(d, abs(fa - fb))
    return d


def similitud_forma(pred: list[float], real: list[float]) -> float:
    """1 - KS: cuánto se parece la *forma* de ambas distribuciones."""
    return 1.0 - ks_dos_muestras(pred, real)


def describe(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "media": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "media": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }
