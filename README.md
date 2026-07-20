# Enjambre 2 · Audiencias Sintéticas LATAM

Segundo producto del portafolio de **Rubicon Lab**. Un enjambre de consumidores
sintéticos, calibrado para mercados latinoamericanos, que reacciona a un
anuncio, concepto de producto o mensaje **antes de lanzarlo al mercado** —
rankea variantes A/B y estima la intención de compra en minutos, no semanas.

> Reutiliza la arquitectura de "enjambre de agentes" del Enjambre 1 (finanzas)
> y la lleva a un nuevo rubro. Ver [`ROADMAP.md`](ROADMAP.md) para el plan por
> fases y `docs/` para el detalle de arquitectura y de la idea.

## Demo rápida (sin API keys)

```bash
python examples/demo.py
```

Corre un pre-test A/B/C de tres mensajes para el mismo producto con una
audiencia sintética LATAM y los rankea por intención de compra. Usa un LLM
**simulado** (determinista), así que corre sin conexión ni llaves — los números
son ilustrativos hasta enchufar un modelo real (Fase 2).

## Uso como librería

```python
from enjambre import Enjambre, MockLLMClient, GeneradorPersonas

enjambre = Enjambre(llm=MockLLMClient(), generador=GeneradorPersonas())
res = enjambre.reaccionar(
    "Café de especialidad en cápsula compostable",
    n_personas=30,
    contexto="lanzamiento en México y Colombia",
)
print(res.media(), res.top_2_box(), res.distribucion())
```

Para usar Claude real:

```python
from enjambre import Enjambre, AnthropicClient
enjambre = Enjambre(llm=AnthropicClient(modelo="claude-sonnet-5"))  # requiere ANTHROPIC_API_KEY
```

## Estructura

```
src/enjambre/
  personas.py      Generador de personas ancladas en segmentos LATAM (+ OCEAN)
  llm.py           Clientes de LLM intercambiables (Mock / Anthropic)
  elicitacion.py   Intención de compra desde texto libre (ExtractorRating, SSR)
  enjambre.py      Orquestador: estímulo -> reacción colectiva -> ranking
  calibracion.py   Registro y correlación predicción vs. realidad (el foso)
examples/demo.py   Demo end-to-end
docs/              Arquitectura y ficha de la idea #1
```

## Estado

Fase 0 (fundaciones) completa: el flujo corre end-to-end con LLM simulado.
Próximo: Fase 1 (personas ancladas en datos reales). Ver `ROADMAP.md`.

## Requisitos

Python 3.10+. La demo no necesita dependencias externas. Para el LLM real:
`pip install -r requirements.txt`.
