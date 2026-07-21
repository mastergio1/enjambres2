"""Runner de calibración reproducible.

Ata todo el arnés de la Fase 3 en una corrida trazable y persistente:
elige LLM + embedder, corre el backtest ciego, puntúa con la doble métrica, y
guarda un reporte JSON con toda la metadata de reproducibilidad (semilla,
versión de anclas, embedder, modelo, hash del dataset). Además registra cada
predicción en la base de calibración (el foso).

CLI:
    python -m enjambre.calibrar --items data/items_backtest.example.json \
        --embedder st --n-agentes 100 --n-items 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .backtest import (
    Backtest,
    GeneradorChile,
    cargar_items,
    reporte,
    scoring,
    seleccion_estratificada,
)
from .calibracion import RegistroCalibracion
from .elicitacion import ANCLAS_CHILE, SSR
from .embeddings import crear_embedder
from .llm import LLMClient, MockLLMClient

SEG_CHILE_DEFAULT = "data/segmentos_chile.example.json"
VERSION_ANCLAS = "ANCLAS_CHILE/v1"


def _sha1(ruta: Path) -> str:
    return hashlib.sha1(Path(ruta).read_bytes()).hexdigest()[:12]


def correr_calibracion(
    items_path: str | Path,
    *,
    seg_path: str | Path = SEG_CHILE_DEFAULT,
    llm: LLMClient | None = None,
    embedder=None,
    nombre_embedder: str | None = None,
    modelo_llm: str = "claude-sonnet-5",
    n_agentes: int = 100,
    n_items: int = 50,
    min_resenas: int = 50,
    semilla: int = 0,
    salida_dir: str | Path = "data/reportes",
    registrar: bool = True,
) -> tuple[dict, list, Path]:
    items = cargar_items(items_path)
    seleccion = seleccion_estratificada(items, n=n_items, min_resenas=min_resenas, semilla=semilla)
    generador = GeneradorChile.desde_json(seg_path, semilla=semilla)

    if embedder is None:
        embedder = crear_embedder(nombre_embedder)
    nombre_embedder = nombre_embedder or getattr(embedder, "nombre", type(embedder).__name__)
    ssr = SSR(embedder=embedder, anclas=ANCLAS_CHILE)

    if llm is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from .llm import AnthropicClient

            llm = AnthropicClient(modelo=modelo_llm)
            nombre_llm = modelo_llm
        else:
            llm = MockLLMClient()
            nombre_llm = "mock (sin señal esperada)"
    else:
        nombre_llm = getattr(llm, "modelo", type(llm).__name__)

    bt = Backtest(llm=llm, generador=generador, ssr=ssr, n_agentes=n_agentes)
    predicciones = bt.correr(seleccion)
    sc = scoring(predicciones)

    reporte_dict = {
        "metadata": {
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "semilla": semilla,
            "n_items": len(seleccion),
            "n_agentes": n_agentes,
            "min_resenas": min_resenas,
            "embedder": nombre_embedder,
            "modelo_llm": nombre_llm,
            "version_anclas": VERSION_ANCLAS,
            "dataset": str(items_path),
            "dataset_sha1": _sha1(Path(items_path)),
            "segmentacion": str(seg_path),
            "gse_nota": "solo AB→C2 (C3/D/E sin verificar, excluidos)",
        },
        "scoring": sc,
        "items": [
            {
                "id": p.item.id,
                "nombre": p.item.nombre,
                "nota_real": p.item.nota_real,
                "nota_predicha": round(p.nota_predicha, 3),
                "banda_real": None,
            }
            for p in predicciones
        ],
    }

    salida = Path(salida_dir)
    salida.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = salida / f"calibracion_{stamp}.json"
    destino.write_text(json.dumps(reporte_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    if registrar:
        reg = RegistroCalibracion()
        caso = Path(items_path).stem
        for p in predicciones:
            idp = reg.registrar_prediccion(caso=caso, estimulo=p.item.nombre, prediccion=p.nota_predicha)
            reg.registrar_real(idp, p.item.nota_real)
        reg.cerrar()

    return reporte_dict, predicciones, destino


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Corre un backtest de calibración reproducible.")
    ap.add_argument("--items", default="data/items_backtest.example.json")
    ap.add_argument("--seg", default=SEG_CHILE_DEFAULT)
    ap.add_argument("--embedder", default=None, help="lexico | st | openai | voyage")
    ap.add_argument("--modelo", default="claude-sonnet-5")
    ap.add_argument("--n-agentes", type=int, default=100)
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--min-resenas", type=int, default=50)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--no-registrar", action="store_true")
    args = ap.parse_args()

    reporte_dict, _, destino = correr_calibracion(
        args.items, seg_path=args.seg, nombre_embedder=args.embedder, modelo_llm=args.modelo,
        n_agentes=args.n_agentes, n_items=args.n_items, min_resenas=args.min_resenas,
        semilla=args.semilla, registrar=not args.no_registrar,
    )
    meta = reporte_dict["metadata"]
    print(f"\n  Embedder: {meta['embedder']}  ·  LLM: {meta['modelo_llm']}  ·  semilla: {meta['semilla']}")
    print(reporte(reporte_dict["scoring"]))
    print(f"\n  Reporte guardado en: {destino}\n")


if __name__ == "__main__":
    _cli()
