"""Clientes de modelo de lenguaje intercambiables.

El enjambre habla con un LLM a través de la interfaz ``LLMClient``. En
desarrollo usamos ``MockLLMClient`` (determinista, sin llaves de API) para
poder correr todo el flujo end-to-end. En producción se enchufa
``AnthropicClient`` u otro proveedor sin tocar el resto del código.
"""
from __future__ import annotations

import hashlib
import os
import time

# Modelos que rechazan el parámetro `temperature` (HTTP 400). Aprendido del
# Enjambre 1: Sonnet 5 lo rechaza; la variabilidad viene del prompt y la semilla.
_PREFIJOS_SIN_TEMPERATURE = ("claude-sonnet-5", "claude-opus-", "claude-fable-")


def _acepta_temperature(modelo: str) -> bool:
    return not any(modelo.startswith(p) for p in _PREFIJOS_SIN_TEMPERATURE)


def reaccion_fallback(prompt: str) -> str:
    """Reacción neutra y determinista para cuando la API no responde.

    Que la simulación NUNCA se caiga por la API (principio del Enjambre 1). Es
    neutra a propósito: el SSR la puntúa ~3, así una falla puntual no sesga el
    resultado. Su uso se cuenta aparte para no ocultarlo.
    """
    return (
        "No tengo una opinión muy marcada sobre esto; ni me atrae ni me molesta "
        "particularmente, lo tendría que ver con más calma."
    )


class LLMClient:
    """Interfaz mínima que cualquier proveedor debe implementar.

    ``es_real`` (lista blanca): un cliente cuenta como LLM real SOLO si lo
    declara explícitamente. Por defecto es False, así cualquier cliente nuevo o
    de prueba se trata como simulación hasta que alguien conscientemente lo
    marque real. Esto protege contra el sesgo de optimismo: el error, si ocurre,
    cae siempre del lado seguro (tratar como validación, no como calibración).
    """

    es_real: bool = False

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Cliente determinista para desarrollo sin conexión ni llaves.

    Deriva la reacción y la intención de compra (1-5) del hash de
    ``sistema + prompt``. Así distintos estímulos y distintas personas
    producen distribuciones distintas *y reproducibles*, suficiente para
    validar el flujo del enjambre antes de gastar tokens reales.
    """

    es_real = False  # determinista: nunca cuenta como LLM real

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

    es_real = True  # único cliente marcado como LLM real

    def __init__(
        self, modelo: str = "claude-sonnet-5", api_key: str | None = None, reintentos: int = 2
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "Instala 'anthropic' (pip install anthropic) para usar AnthropicClient."
            ) from exc
        import anthropic

        self.modelo = modelo
        self.reintentos = reintentos
        self.stats = {"api": 0, "fallback": 0}  # trazabilidad de la fuente de cada reacción
        self._cliente = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:  # pragma: no cover
        params = dict(
            model=self.modelo,
            max_tokens=400,
            system=sistema,
            messages=[{"role": "user", "content": prompt}],
        )
        # No enviar temperature a modelos que la rechazan (evita HTTP 400).
        if temperatura is not None and _acepta_temperature(self.modelo):
            params["temperature"] = temperatura

        for intento in range(self.reintentos + 1):
            try:
                msg = self._cliente.messages.create(**params)
                texto = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                if texto.strip():
                    self.stats["api"] += 1
                    return texto
            except Exception:  # noqa: BLE001 - reintenta y, si todo falla, hace fallback
                if intento < self.reintentos:
                    time.sleep(min(2**intento, 4))
        # La simulación nunca se cae por la API.
        self.stats["fallback"] += 1
        return reaccion_fallback(prompt)


class LLMCache(LLMClient):
    """Envuelve cualquier cliente y cachea por (modelo, sistema, prompt).

    Repetir un pre-test o una demo no vuelve a gastar tokens. En memoria por
    defecto; espeja ``es_real`` del cliente base para no romper la deducción del
    tipo de corrida.
    """

    def __init__(self, base: LLMClient) -> None:
        self.base = base
        self.es_real = getattr(base, "es_real", False)
        self._cache: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def _clave(self, sistema: str, prompt: str) -> str:
        modelo = getattr(self.base, "modelo", type(self.base).__name__)
        return hashlib.sha256(f"{modelo}\x00{sistema}\x00{prompt}".encode("utf-8")).hexdigest()

    def completar(self, sistema: str, prompt: str, *, temperatura: float = 0.9) -> str:
        clave = self._clave(sistema, prompt)
        if clave in self._cache:
            self.hits += 1
            return self._cache[clave]
        self.misses += 1
        texto = self.base.completar(sistema, prompt, temperatura=temperatura)
        self._cache[clave] = texto
        return texto
