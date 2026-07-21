"""Tests del blindaje: tipo de corrida deducido y dataset ciego (Tareas 1 y 6)."""
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import MockLLMClient, TipoCorrida, deducir_tipo  # noqa: E402
from enjambre.dataset import (  # noqa: E402
    EstimuloCiego,
    cargar_dataset_ciego,
    cargar_dataset_ejemplo,
)

DATOS = RAIZ / "data"


class LLMFalsoReal(MockLLMClient):
    es_real = True


def _escribir(tmp_path, est, clave):
    pe = tmp_path / "e.json"
    pc = tmp_path / "c.json"
    pe.write_text(json.dumps(est), encoding="utf-8")
    pc.write_text(json.dumps(clave), encoding="utf-8")
    return pe, pc


_EST_OK = [{"id": "a", "tipo": "app", "nombre": "N", "categoria": "C", "descripcion": "d"}]
_CLAVE_OK = [{"id": "a", "valoracion_real": 4.0, "fuente": "Kaggle GPlay 2023",
              "fecha": "2023-01-01", "n_muestras": 100}]


# --- Tarea 1: deducción no falsificable ---
def test_mock_siempre_es_validacion():
    ds = cargar_dataset_ejemplo(DATOS / "items_backtest.example.json")
    assert deducir_tipo(MockLLMClient(), ds)["tipo"] is TipoCorrida.VALIDACION_ARNES


def test_calibracion_real_requiere_llm_y_dataset_reales(tmp_path):
    pe, pc = _escribir(tmp_path, _EST_OK, _CLAVE_OK)
    ds_real = cargar_dataset_ciego(pe, pc)
    assert ds_real.es_real is True
    # real + real ⇒ calibración real
    assert deducir_tipo(LLMFalsoReal(), ds_real)["tipo"] is TipoCorrida.CALIBRACION_REAL
    # LLM real pero dataset de ejemplo ⇒ sigue siendo validación
    ds_ej = cargar_dataset_ejemplo(DATOS / "items_backtest.example.json")
    assert deducir_tipo(LLMFalsoReal(), ds_ej)["tipo"] is TipoCorrida.VALIDACION_ARNES


def test_declarar_tipo_contradictorio_se_ignora_y_avisa():
    ds = cargar_dataset_ejemplo(DATOS / "items_backtest.example.json")
    res = deducir_tipo(MockLLMClient(), ds, tipo_declarado="CALIBRACION_REAL")
    assert res["tipo"] is TipoCorrida.VALIDACION_ARNES
    assert res["avisos"] and "ignoró" in res["avisos"][0].lower()


# --- Tarea 6: dataset ciego estructural y validado ---
def test_estimulo_ciego_no_puede_contener_nota():
    campos = set(EstimuloCiego.__dataclass_fields__)
    assert not (campos & {"nota", "nota_real", "valoracion", "valoracion_real"})


def test_ciego_rechaza_nota_en_estimulo(tmp_path):
    pe, pc = _escribir(tmp_path, [{**_EST_OK[0], "nota_real": 4.5}], _CLAVE_OK)
    with pytest.raises(ValueError):
        cargar_dataset_ciego(pe, pc)


def test_ciego_rechaza_ids_desalineados(tmp_path):
    pe, pc = _escribir(tmp_path, _EST_OK, [{**_CLAVE_OK[0], "id": "z"}])
    with pytest.raises(ValueError):
        cargar_dataset_ciego(pe, pc)


def test_ciego_rechaza_sin_procedencia(tmp_path):
    pe, pc = _escribir(tmp_path, _EST_OK, [{"id": "a", "valoracion_real": 4.0}])
    with pytest.raises(ValueError):
        cargar_dataset_ciego(pe, pc)


def test_ejemplo_nunca_es_real():
    assert cargar_dataset_ejemplo(DATOS / "items_backtest.example.json").es_real is False
    # incluso los archivos ciegos de ejemplo (procedencia sintética) no son reales
    assert cargar_dataset_ciego(
        DATOS / "estimulos_ciegos.example.json",
        DATOS / "clave_resultados.example.json",
    ).es_real is False
