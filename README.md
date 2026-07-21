# Enjambre 2 · Audiencias Sintéticas LATAM

Segundo producto del portafolio de **Rubicon Lab**. Un enjambre de consumidores
sintéticos, calibrado para mercados latinoamericanos, que reacciona a un
anuncio, concepto de producto o mensaje **antes de lanzarlo al mercado** —
rankea variantes A/B y estima la intención de compra en minutos, no semanas.

> Reutiliza la arquitectura de "enjambre de agentes" del Enjambre 1 (finanzas)
> y la lleva a un nuevo rubro. Ver [`ROADMAP.md`](ROADMAP.md) para el plan por
> fases y `docs/` para el detalle de arquitectura y de la idea.

## Interfaz web (sube y reacciona)

```bash
python -m enjambre.servidor --port 8000   # abre http://localhost:8000
```

Pega tu producto y sus variantes de mensaje, elige la audiencia (LATAM o Chile)
y el tamaño, y recibe el ranking por intención de compra + distribución +
reacciones de ejemplo, con export a JSON. Corre sin dependencias (stdlib) y sin
llaves (LLM simulado); con `ANTHROPIC_API_KEY` usa Claude real.

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
  personas.py      Personas ancladas en segmentos LATAM (demografía + condicionales + OCEAN + intereses)
  conocimiento.py  Capa RAG: aterriza personas en reseñas/tickets reales (recuperador TF-IDF)
  fuentes.py       Adaptadores de fuentes externas al corpus RAG (p. ej. opiniones de X)
  llm.py           Clientes de LLM intercambiables (Mock / Anthropic)
  elicitacion.py   Intención de compra desde texto libre: ExtractorRating y método SSR (+ Embedder)
  enjambre.py      Orquestador: estímulo -> reacción colectiva (con RAG) -> ranking
  backtest.py      Arnés de calibración ciego (persona Chile + SSR + estratificación)
  embeddings.py    Embedders semánticos reales (SentenceTransformers/OpenAI/Voyage) + fábrica
  calibrar.py      Runner reproducible: corre el backtest y persiste reporte con metadata
  metricas.py      Correlación, forma de distribución (KS), MAE, baseline tonto
  api.py           Lógica de la interfaz "sube y reacciona" (independiente de HTTP)
  servidor.py      Servidor stdlib que expone la API y sirve la web UI
  web/index.html   Interfaz web (responsive, claro/oscuro)
  validacion.py    ¿La audiencia sintética reproduce la distribución objetivo?
  calibracion.py   Registro y correlación predicción vs. realidad (el foso)
examples/demo.py                    Demo end-to-end (Fase 0)
examples/demo_fase1.py              Personas ancladas + validación + RAG (Fase 1)
examples/demo_fase2.py              Método SSR + Claude real opcional (Fase 2)
examples/demo_fase3.py              Backtest de calibración ciego, foco Chile (Fase 3)
examples/validar_arnes.py           Valida el instrumento — NO es calibración real (offline)
data/                    Segmentación, reseñas e ítems de ejemplo (reemplazar por datos del cliente)
docs/                    Contexto y casos de uso, arquitectura, calibración e idea #1
```

## Estado

Fases 0–4 completas: personas ancladas en datos demográficos reales (LATAM y
Chile) y aterrizadas con RAG; medición de intención con el método **SSR**
(embedders semánticos reales enchufables); **arnés de backtest ciego** con doble
métrica y controles anti-autoengaño; **runner de calibración reproducible**; e
**interfaz web "sube y reacciona"** (API + UI) que cualquiera puede usar sin
código. Claude real conectable con `ANTHROPIC_API_KEY`. Próximo: Fase 5 (piloto
con cliente). Ver `ROADMAP.md`. Panorama del producto: `docs/contexto.md`;
método de calibración: `docs/calibracion.md`.

## Correr la calibración real

```bash
pip install sentence-transformers          # embedder semántico local (sin llave)
ANTHROPIC_API_KEY=... python -m enjambre.calibrar \
    --embedder st --items data/tu_dataset_con_notas.json --n-agentes 100 --n-items 50
```

## Requisitos

Python 3.10+. La demo no necesita dependencias externas. Para el LLM real:
`pip install -r requirements.txt`.
