"""Servidor de la interfaz "sube y reacciona" (stdlib, sin dependencias).

Expone la lógica de ``api.py`` por HTTP y sirve la web UI. Se eligió la stdlib
(``http.server``) para que la interfaz corra en cualquier entorno con solo
Python. En producción se puede migrar a FastAPI reutilizando ``api.py`` intacto.

    python -m enjambre.servidor --port 8000
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import api

WEB = Path(__file__).resolve().parent / "web"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silencioso
        pass

    def _enviar_json(self, codigo: int, obj: dict) -> None:
        cuerpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _enviar_archivo(self, ruta: Path, tipo: str) -> None:
        if not ruta.exists():
            return self._enviar_json(404, {"error": "no encontrado"})
        datos = ruta.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            return self._enviar_archivo(WEB / "index.html", "text/html; charset=utf-8")
        if self.path == "/api/opciones":
            return self._enviar_json(200, api.opciones())
        if self.path == "/api/salud":
            return self._enviar_json(200, {"ok": True})
        self._enviar_json(404, {"error": "no encontrado"})

    def do_POST(self) -> None:
        if self.path != "/api/pretest":
            return self._enviar_json(404, {"error": "no encontrado"})
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            cuerpo = self.rfile.read(n) if n else b"{}"
            payload = json.loads(cuerpo or b"{}")
            self._enviar_json(200, api.ejecutar_pretest(payload))
        except ValueError as e:
            self._enviar_json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._enviar_json(500, {"error": f"error interno: {e}"})


def crear_servidor(puerto: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("0.0.0.0", puerto), Handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interfaz web del Enjambre 2.")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    servidor = crear_servidor(args.port)
    modo = api.opciones()["llm"]
    print(f"  Enjambre 2 · interfaz en http://localhost:{args.port}  (LLM: {modo})")
    print("  Ctrl+C para parar.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        servidor.shutdown()


if __name__ == "__main__":
    main()
