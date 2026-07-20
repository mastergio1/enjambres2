"""Clientes de modelo de lenguaje intercambiables.

El enjambre habla con un LLM a través de la interfaz ``LLMClient``. En
desarrollo usamos ``MockLLMClient`` (determinista, sin llaves de API) para
poder correr todo el flujo end-to-end. En producción se enchufa
``AnthropicClient`` u otro proveedor sin tocar el resto del código.
"""
from __future__ import annotations

import hashlib
import os


class LLMClient:
    """Interfaz mínima que cualquier proveedor debe implementar."""

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Cliente determinista para desarrollo sin conexión ni llaves.

    Deriva la reacción y la intención de compra (1-5) del hash de
    ``sistema + prompt``. Así distintos estímulos y distintas personas
    producen distribuciones distintas *y reproducibles*, suficiente para
    validar el flujo del enjambre antes de gastar tokens reales.
    """

    _FRASES = [
        "no me dice nada, lo paso de largo",
        "me genera dudas, no termino de engancharme",
        "me parece interesante, lo consideraría",
        "me gusta bastante, seguramente lo probaría",
        "me encanta, lo compraría sin pensarlo",
    ]

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:
        semilla = int(hashlib.sha256((sistema + "||" + prompt).encode("utf-8")).hexdigest(), 16)
        rating = 1 + (semilla % 5)
        frase = self._FRASES[rating - 1]
        return f"Reacción: {frase}. Intención de compra: {rating}/5."


class AnthropicClient(LLMClient):
    """Cliente real contra la API de Claude. Requiere ``anthropic`` y ANTHROPIC_API_KEY."""

    def __init__(self, modelo: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "Instala 'anthropic' (pip install anthropic) para usar AnthropicClient."
            ) from exc
        import anthropic

        self.modelo = modelo
        self._cliente = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:  # pragma: no cover
        msg = self._cliente.messages.create(
            model=self.modelo,
            max_tokens=400,
            temperature=temperatura,
            system=sistema,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
