"""Runner de calibración reproducible, con tipo de corrida deducido.

Ata el arnés en una corrida trazable: deduce si es VALIDACION_ARNES o
CALIBRACION_REAL (del entorno, no de lo declarado), corre el backtest ciego,
puntúa, y persiste un reporte cuyo NOMBRE y CONTENIDO dejan claro qué tipo es.
Registra cada predicción en la base con su tipo.

CLI:
    # validación del arnés (dataset de ejemplo)
    python -m enjambre.calibrar --items data/items_backtest.example.json --embedder lexico
    # calibración real (dos archivos separados, con LLM real)
    ANTHROPIC_API_KEY=... python -m enjambre.calibrar \
        --estimulos data/estimulos.json --clave data/clave.json --embedder st
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .backtest import Backtest, GeneradorChile, reporte, scoring
from .calibracion import RegistroCalibracion
from .corrida import TipoCorrida, deducir_tipo
from .dataset import DatasetCalibracion, cargar_dataset_ciego, cargar_dataset_ejemplo, seleccion_estratificada
from .elicitacion import ANCLAS_CHILE, SSR
from .embeddings import crear_embedder
from .llm import LLMClient, MockLLMClient

SEG_CHILE_DEFAULT = "data/segmentos_chile.example.json"
VERSION_ANCLAS = "ANCLAS_CHILE/v1"

_ADVERTENCIA_VALIDACION = (
    "⚠ VALIDACIÓN DEL ARNÉS — este número mide el INSTRUMENTO (si detecta señal "
    "cuando la hay), NO el desempeño predictivo del producto. NO usar en material "
    "comercial ni citar como calibración frente a un cliente."
)

_PREFIJO = {
    TipoCorrida.VALIDACION_ARNES: "VALIDACION-ARNES",
    TipoCorrida.CALIBRACION_REAL: "CALIBRACION-REAL",
}


def correr_calibracion(
    dataset: DatasetCalibracion,
    *,
    seg_path: str | Path = SEG_CHILE_DEFAULT,
    llm: LLMClient | None = None,
    embedder=None,
    nombre_embedder: str | None = None,
    modelo_llm: str = "claude-sonnet-5",
    n_agentes: int = 100,
    n_items: int = 50,
    min_muestras: int = 50,
    semilla: int = 0,
    salida_dir: str | Path = "data/reportes",
    registrar: bool = True,
    tipo_declarado: TipoCorrida | str | None = None,
) -> tuple[dict, list, Path]:
    seleccion = seleccion_estratificada(dataset, n=n_items, min_muestras=min_muestras, semilla=semilla)
    generador = GeneradorChile.desde_json(seg_path, semilla=semilla)

    if embedder is None:
        embedder = crear_embedder(nombre_embedder)
    nombre_embedder = nombre_embedder or getattr(embedder, "nombre", type(embedder).__name__)
    ssr = SSR(embedder=embedder, anclas=ANCLAS_CHILE)

    if llm is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from .llm import AnthropicClient

            llm = AnthropicClient(modelo=modelo_llm)
        else:
            llm = MockLLMClient()
    nombre_llm = getattr(llm, "modelo", type(llm).__name__)

    # --- Tipo de corrida DEDUCIDO del entorno (no declarado) ---
    ded = deducir_tipo(llm, dataset, tipo_declarado)
    tipo: TipoCorrida = ded["tipo"]

    bt = Backtest(llm=llm, generador=generador, ssr=ssr, n_agentes=n_agentes)
    predicciones = bt.correr(seleccion)
    sc = scoring(predicciones, dataset)

    reporte_dict: dict = {}
    if tipo is TipoCorrida.VALIDACION_ARNES:
        reporte_dict["_ADVERTENCIA"] = _ADVERTENCIA_VALIDACION
    reporte_dict.update({
        "tipo_corrida": tipo.value,
        "metadata": {
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "cliente_llm": nombre_llm,
            "llm_es_real": bool(getattr(llm, "es_real", False)),
            "embedder": nombre_embedder,
            "version_anclas": VERSION_ANCLAS,
            "dataset_origen": dataset.origen,
            "dataset_es_real": dataset.es_real,
            "sha1_estimulos": dataset.sha1_estimulos,
            "sha1_clave": dataset.sha1_clave,
            "semilla": semilla,
            "n_items": len(seleccion),
            "n_agentes": n_agentes,
            "min_muestras": min_muestras,
            "gse_nota": "solo AB→C2 (C3/D/E sin verificar, excluidos)",
            "condiciones_realidad": ded["condiciones"],
            "avisos": ded["avisos"],
        },
        "scoring": sc,
        "items": [
            {
                "id": p.estimulo.id,
                "nombre": p.estimulo.nombre,
                "nota_real": dataset.resultado(p.estimulo.id),
                "nota_predicha": round(p.nota_predicha, 3),
                "procedencia": {
                    "fuente": dataset.procedencia(p.estimulo.id).fuente,
                    "fecha": dataset.procedencia(p.estimulo.id).fecha,
                },
            }
            for p in predicciones
        ],
    })

    salida = Path(salida_dir)
    salida.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = salida / f"{_PREFIJO[tipo]}_{stamp}.json"
    destino.write_text(json.dumps(reporte_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    if registrar:
        reg = RegistroCalibracion()
        for p in predicciones:
            idp = reg.registrar_prediccion(
                caso=dataset.origen, estimulo=p.estimulo.nombre,
                prediccion=p.nota_predicha, tipo=tipo,
            )
            reg.registrar_real(idp, dataset.resultado(p.estimulo.id))
        reg.cerrar()

    return reporte_dict, predicciones, destino


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Corre un backtest de calibración reproducible.")
    ap.add_argument("--items", default=None, help="dataset de ejemplo (notas inline) → validación")
    ap.add_argument("--estimulos", default=None, help="estímulos ciegos (sin notas) → real")
    ap.add_argument("--clave", default=None, help="clave de resultados (con procedencia)")
    ap.add_argument("--seg", default=SEG_CHILE_DEFAULT)
    ap.add_argument("--embedder", default=None, help="lexico | st | openai | voyage")
    ap.add_argument("--modelo", default="claude-sonnet-5")
    ap.add_argument("--n-agentes", type=int, default=100)
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--min-muestras", type=int, default=50)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--no-registrar", action="store_true")
    args = ap.parse_args()

    if args.estimulos and args.clave:
        dataset = cargar_dataset_ciego(args.estimulos, args.clave)
    elif args.items:
        dataset = cargar_dataset_ejemplo(args.items)
    else:
        ap.error("Da --items (ejemplo) o --estimulos + --clave (real).")

    reporte_dict, _, destino = correr_calibracion(
        dataset, seg_path=args.seg, nombre_embedder=args.embedder, modelo_llm=args.modelo,
        n_agentes=args.n_agentes, n_items=args.n_items, min_muestras=args.min_muestras,
        semilla=args.semilla, registrar=not args.no_registrar,
    )
    meta = reporte_dict["metadata"]
    print(f"\n  TIPO DE CORRIDA: {reporte_dict['tipo_corrida']}")
    if "_ADVERTENCIA" in reporte_dict:
        print("  " + reporte_dict["_ADVERTENCIA"])
    for aviso in meta["avisos"]:
        print("  ⚠ " + aviso)
    print(f"  LLM: {meta['cliente_llm']} (real={meta['llm_es_real']})  ·  "
          f"embedder: {meta['embedder']}  ·  dataset: {meta['dataset_origen']} (real={meta['dataset_es_real']})")
    print(reporte(reporte_dict["scoring"]))
    print(f"\n  Reporte guardado en: {destino}\n")


if __name__ == "__main__":
    _cli()
