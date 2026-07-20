"""Registro y calibración: el verdadero foso del producto.

Cada predicción del enjambre se guarda. Cuando llega el resultado real
(conversión, ventas, engagement), se registra y se calcula la correlación
histórica. Ese track record —"sabemos dónde acertamos y dónde no"— es lo
que convierte un juguete en una herramienta creíble.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

RUTA_DB_DEFAULT = Path("data/calibracion.db")


class RegistroCalibracion:
    def __init__(self, ruta: Path | str = RUTA_DB_DEFAULT) -> None:
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.ruta)
        self._crear()

    def _crear(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caso TEXT NOT NULL,
                estimulo TEXT NOT NULL,
                prediccion REAL NOT NULL,
                real REAL,
                creado TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def registrar_prediccion(self, caso: str, estimulo: str, prediccion: float) -> int:
        cur = self._conn.execute(
            "INSERT INTO predicciones (caso, estimulo, prediccion) VALUES (?, ?, ?)",
            (caso, estimulo, prediccion),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def registrar_real(self, id_prediccion: int, valor_real: float) -> None:
        self._conn.execute(
            "UPDATE predicciones SET real = ? WHERE id = ?", (valor_real, id_prediccion)
        )
        self._conn.commit()

    def pares_cerrados(self) -> list[tuple[float, float]]:
        cur = self._conn.execute(
            "SELECT prediccion, real FROM predicciones WHERE real IS NOT NULL"
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def correlacion(self) -> float | None:
        """Pearson entre predicción y realidad sobre los casos ya cerrados."""
        pares = self.pares_cerrados()
        if len(pares) < 2:
            return None
        xs, ys = zip(*pares)
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in pares)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx == 0 or vy == 0:
            return None
        return cov / (vx**0.5 * vy**0.5)

    def cerrar(self) -> None:
        self._conn.close()
