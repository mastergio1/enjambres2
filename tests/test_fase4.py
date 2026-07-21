"""Tests de la Fase 4: lógica de la API y servidor HTTP."""
import json
import sys
import threading
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from enjambre import api  # noqa: E402
from enjambre.servidor import crear_servidor  # noqa: E402


def test_opciones_tiene_mercados_y_metodos():
    op = api.opciones()
    assert "latam" in op["mercados"] and "chile" in op["mercados"]
    assert "rapido" in op["metodos"] and "semantico" in op["metodos"]
    assert op["llm"] in ("claude", "simulado")


def test_pretest_rankea_variantes():
    data = api.ejecutar_pretest({
        "producto": "Café en cápsula compostable",
        "variantes": [
            {"nombre": "A", "texto": "Café premium, sabor intenso."},
            {"nombre": "B", "texto": "El café más barato y rendidor."},
        ],
        "n_personas": 30, "mercado": "latam", "metodo": "rapido",
    })
    assert len(data["resultados"]) == 2
    assert data["ganadora"] == data["resultados"][0]["nombre"]
    # ranking descendente por media
    medias = [r["media"] for r in data["resultados"]]
    assert medias == sorted(medias, reverse=True)
    r0 = data["resultados"][0]
    assert 1 <= r0["media"] <= 5 and 0 <= r0["top_2_box"] <= 100
    assert sum(r0["distribucion"].values()) >= 1


def test_pretest_mercado_chile_y_semantico():
    data = api.ejecutar_pretest({
        "producto": "App de ahorro",
        "variantes": {"Única": "Controla tus gastos y ahorra."},
        "n_personas": 20, "mercado": "chile", "metodo": "semantico",
    })
    assert data["mercado"] == "chile" and data["metodo"] == "semantico"
    assert len(data["resultados"]) == 1


def test_pretest_validaciones():
    import pytest
    with pytest.raises(ValueError):
        api.ejecutar_pretest({"producto": "", "variantes": [{"texto": "x"}]})
    with pytest.raises(ValueError):
        api.ejecutar_pretest({"producto": "algo", "variantes": []})


def test_servidor_http_extremo_a_extremo():
    servidor = crear_servidor(0)  # puerto libre
    puerto = servidor.server_address[1]
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        base = f"http://127.0.0.1:{puerto}"
        # UI
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert r.status == 200 and b"Enjambre 2" in r.read()
        # opciones
        with urllib.request.urlopen(base + "/api/opciones", timeout=5) as r:
            assert "mercados" in json.loads(r.read())
        # pretest
        cuerpo = json.dumps({
            "producto": "Bebida energética natural",
            "variantes": [{"nombre": "A", "texto": "Energía sin químicos"},
                          {"nombre": "B", "texto": "La más potente del mercado"}],
            "n_personas": 15,
        }).encode()
        req = urllib.request.Request(base + "/api/pretest", data=cuerpo,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            assert r.status == 200 and len(data["resultados"]) == 2 and data["ganadora"]
    finally:
        servidor.shutdown()
        servidor.server_close()
