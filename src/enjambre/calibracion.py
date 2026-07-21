"""Registro y calibración: el verdadero foso del producto.

Cada predicción del enjambre se guarda con su TIPO DE CORRIDA. Es el activo
defendible del negocio (histórico predicción-vs-realidad) y no puede
contaminarse: las corridas de validación del arnés y las de calibración real
viven en la misma tabla pero jamás se mezclan en un cálculo.

Reglas (Tarea 2):
- ``tipo`` es columna obligatoria; no hay registro sin tipo.
- No existe una consulta "traer todo": toda lectura filtra por tipo.
- El track record (correlación histórica) opera SOLO sobre CALIBRACION_REAL y
  falla si detecta cualquier otro tipo en el conjunto que va a agregar.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .corrida import TipoCorrida

RUTA_DB_DEFAULT = Path("data/calibracion.db")
_TIPOS_VALIDOS = {t.value for t in TipoCorrida}


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
                tipo TEXT NOT NULL,
                caso TEXT NOT NULL,
                estimulo TEXT NOT NULL,
                prediccion REAL NOT NULL,
                real REAL,
                creado TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migración defensiva: si una DB vieja no tiene la columna 'tipo', añadirla.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(predicciones)")}
        if "tipo" not in cols:
            self._conn.execute("ALTER TABLE predicciones ADD COLUMN tipo TEXT")
        self._conn.commit()

    @staticmethod
    def _valida_tipo(tipo: TipoCorrida | str) -> str:
        valor = tipo.value if isinstance(tipo, TipoCorrida) else str(tipo)
        if valor not in _TIPOS_VALIDOS:
            raise ValueError(f"tipo de corrida inválido: {valor!r}. Debe ser uno de {_TIPOS_VALIDOS}.")
        return valor

    def registrar_prediccion(
        self, caso: str, estimulo: str, prediccion: float, tipo: TipoCorrida | str
    ) -> int:
        """Registra una predicción. ``tipo`` es OBLIGATORIO (sin valor por defecto)."""
        valor_tipo = self._valida_tipo(tipo)
        cur = self._conn.execute(
            "INSERT INTO predicciones (tipo, caso, estimulo, prediccion) VALUES (?, ?, ?, ?)",
            (valor_tipo, caso, estimulo, prediccion),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def registrar_real(self, id_prediccion: int, valor_real: float) -> None:
        self._conn.execute(
            "UPDATE predicciones SET real = ? WHERE id = ?", (valor_real, id_prediccion)
        )
        self._conn.commit()

    def contar(self, tipo: TipoCorrida | str) -> int:
        valor = self._valida_tipo(tipo)
        cur = self._conn.execute("SELECT COUNT(*) FROM predicciones WHERE tipo = ?", (valor,))
        return int(cur.fetchone()[0])

    def pares_cerrados(self, tipo: TipoCorrida | str) -> list[tuple[float, float]]:
        """Pares (predicción, real) de un tipo concreto. No existe versión 'de todo'."""
        valor = self._valida_tipo(tipo)
        cur = self._conn.execute(
            "SELECT prediccion, real FROM predicciones WHERE real IS NOT NULL AND tipo = ?",
            (valor,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def correlacion_real(self) -> float | None:
        """Track record del negocio: correlación histórica SOLO sobre CALIBRACION_REAL.

        Falla si en el conjunto de casos cerrados marcados como reales se cuela
        cualquier otro tipo (seguro contra contaminación silenciosa).
        """
        cur = self._conn.execute(
            "SELECT prediccion, real, tipo FROM predicciones WHERE real IS NOT NULL "
            "AND tipo = ?",
            (TipoCorrida.CALIBRACION_REAL.value,),
        )
        filas = cur.fetchall()
        intrusos = [t for _, _, t in filas if t != TipoCorrida.CALIBRACION_REAL.value]
        if intrusos:
            raise ValueError(
                f"Contaminación detectada: {len(intrusos)} registro(s) no reales en el "
                "conjunto de calibración real. Aborto el cálculo del track record."
            )
        pares = [(p, r) for p, r, _ in filas]
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
