# Enjambre 2 — Visión general (producto + técnica)

Documento único que resume **de qué va el proyecto, para qué está creado, sus
casos de uso y cómo está construido en código**. Para el detalle: producto en
[`contexto.md`](contexto.md), arquitectura en [`arquitectura.md`](arquitectura.md),
calibración en [`calibracion.md`](calibracion.md), plan en [`../ROADMAP.md`](../ROADMAP.md).

---

## 1. De qué va y para qué está creado

**Enjambre 2** es el **segundo producto del portafolio de Rubicon Lab**. Es un
motor de **audiencias sintéticas**: un enjambre de agentes de IA que actúan como
**consumidores reales latinoamericanos**. Le muestras un producto, un anuncio o
un mensaje, y el enjambre **reacciona como lo haría tu público objetivo** — te
dice qué opina, cuánto le interesa y qué variante prefiere.

**El problema que resuelve.** Validar un producto o una campaña con
investigación de mercado tradicional cuesta **miles de dólares y semanas**
(focus groups, encuestas, paneles). Enjambre 2 da una **primera lectura barata y
rápida** para descartar ideas débiles *antes* de gastar dinero real.

**Por qué existe / el foso.** Reutiliza la arquitectura de "enjambre de agentes"
del Enjambre 1 (finanzas) y la lleva a un rubro nuevo (investigación de mercado).
Su ventaja defendible es triple:

1. **Foco LATAM / español** — hueco que los competidores anglosajones no atacan.
2. **Calibración honesta** — cada predicción se compara contra el resultado real.
3. **Posicionamiento honesto** — se vende como *early-warning* y filtro de
   priorización, **no como oráculo**.

> No reemplaza al cliente real en decisiones de alto riesgo. Sirve para reducir
> incertidumbre y priorizar rápido y barato.

---

## 2. Para quién es

| Cliente | Qué gana |
|---------|----------|
| **PYMEs y emprendedores** | Validan ideas sin pagar research caro |
| **Agencias y equipos de marketing** | Pre-testean campañas antes del gasto en medios |
| **Áreas de producto** | Priorizan features y mensajes por segmento |
| **Marcas de consumo (CPG)** | Prueban packaging, naming, claims y precios |

---

## 3. Casos de uso

- **Pre-test de anuncios** — 3 versiones de un post; el enjambre las rankea y
  lanzas solo la ganadora.
- **Validación de concepto** — ¿"café en cápsula compostable" tiene tracción?
  Intención de compra + objeciones por segmento.
- **Naming y pricing** — screening rápido de nombres o puntos de precio.
- **Testeo de mensajes** — descubrir que "sostenibilidad" mueve al NSE alto pero
  no al bajo.
- **Priorización de features** — ordenar el roadmap por reacción de segmentos
  antes de programar.
- **Localización cultural** — cómo cae el mismo mensaje en México vs Argentina vs
  Chile.

---

## 4. Cómo está creado (lenguaje técnico)

**Stack.** Python 3.10+. El **núcleo corre solo con la stdlib** (sin instalar
nada, sin llaves). Las dependencias son **opcionales y por capa**: `anthropic`
(LLM real), `sentence-transformers` / `openai` / `voyageai` (embeddings
semánticos). Suite de tests en verde (35 casos).

**Diseño clave: todo desacoplado por interfaces**, para desarrollar y testear
offline y enchufar lo real sin reescribir.

### El pipeline de un pre-test

```
estímulo (producto + variantes)
   │
   ▼
GeneradorPersonas ──► muestrea la audiencia de una distribución demográfica real
   │                   (país, edad, NSE, urbanidad + OCEAN + intereses)
   ▼
BaseConocimiento (RAG) ──► aterriza cada persona en reseñas/tickets reales
   │
   ▼
Enjambre (orquestador) ──► cada persona reacciona vía LLMClient
   │                        (system prompt = su perfil)
   ▼
elicitador (SSR / ExtractorRating) ──► convierte el texto en intención 1-5
   │
   ▼
Resultado ──► media, top-2-box, dispersión, distribución, ranking A/B/n
```

### Los módulos (paquete `src/enjambre/`)

| Módulo | Qué hace | Piezas clave |
|--------|----------|--------------|
| `llm.py` | Abstracción del modelo | `LLMClient` (interfaz), `MockLLMClient` (dev, determinista), `AnthropicClient` (Claude) |
| `personas.py` | Audiencia LATAM anclada | `Segmentacion.desde_json`, `GeneradorPersonas`, distribuciones **condicionales** (edad\|país) y **priors OCEAN** por segmento |
| `conocimiento.py` | Capa RAG | `BaseConocimiento`, `RecuperadorTFIDF` (stdlib, insensible a acentos, boost por tags) |
| `elicitacion.py` | Texto → intención | `SSR` (similitud semántica vs anclas Likert), `ExtractorRating`, `Embedder` / `EmbedderLexico` |
| `embeddings.py` | Embedders reales | `crear_embedder("st"/"openai"/"voyage")` con imports protegidos |
| `enjambre.py` | Orquestador | `Enjambre.reaccionar()` y `.comparar()`, `Resultado` (métricas) |
| `backtest.py` | Calibración ciega (Chile) | `GeneradorChile` (Censo 2024 + GSE AIM), `Backtest`, `scoring` |
| `metricas.py` | Scoring | Pearson, KS (forma de distribución), MAE, baseline tonto |
| `calibrar.py` | Runner reproducible | `correr_calibracion` + CLI, persiste reporte con metadata |
| `calibracion.py` | El "foso" | `RegistroCalibracion` (SQLite): predicción vs. realidad |
| `validacion.py` | Sanidad de la audiencia | ¿La audiencia sintética reproduce la distribución objetivo? |
| `api.py` + `servidor.py` + `web/` | Interfaz | Lógica separada del transporte HTTP (stdlib), UI web |

### Decisiones técnicas que importan

- **`LLMClient` intercambiable** → desarrollas y testeas con `MockLLMClient`
  (determinista, basado en hash) sin gastar tokens; en producción cambias a
  `AnthropicClient` sin tocar el resto.
- **Método SSR** → no se le pide un número al modelo (eso aplasta la respuesta
  al centro); reacciona en texto libre y se mide por **similitud coseno contra
  5 anclas Likert** + softmax → intención esperada continua. Las anclas se
  **cachean** para no gastar una llamada de embeddings por agente.
- **Elicitación como estrategia** → `Enjambre` acepta cualquier objeto con
  `.intencion(texto)` e `.instruccion_prompt()`, así SSR y ExtractorRating son
  enchufables.
- **Backtest ciego + doble métrica** → el agente **nunca ve la nota real**; se
  puntúa con correlación **y** forma de distribución, más **controles
  anti-autoengaño** (baseline tonto, dispersión, homogeneidad). Prueba honesta:
  un lector ciego da r≈0; un lector que comprende da r≈0.72.
- **Reproducibilidad** → semillas fijas en todo; el runner guarda versión de
  anclas, embedder, modelo y hash del dataset.
- **API sin framework** → `api.py` (lógica pura, testeable) separada de
  `servidor.py` (stdlib `http.server`), migrable a FastAPI sin reescribir.

### Cómo se corre

```bash
python -m enjambre.servidor                      # interfaz web (http://localhost:8000)
python examples/demo.py                           # pre-test A/B/C offline
python examples/validar_arnes.py                  # valida el instrumento (no es calibración real)
python -m enjambre.calibrar --embedder st --items <dataset_con_notas.json>  # calibración real
```

Con LLM simulado corre sin llaves (números ilustrativos). Con `ANTHROPIC_API_KEY`
usa Claude real.

---

## 5. Estado y límites

**Estado:** Fases 0–4 completas — personas ancladas en datos reales (LATAM y
Chile), RAG, medición SSR, arnés de calibración ciego con doble métrica, runner
reproducible e interfaz web usable sin código. Siguiente: piloto con cliente.

**Límites honestos (para vender sin humo):**
- La fidelidad **cae en productos genuinamente novedosos** (correlaciones tan
  bajas como ~0.3); por eso se posiciona como early-warning, no oráculo.
- Riesgo de **homogeneidad** de los agentes → se ancla en demografía real y se
  valida la varianza, no solo la media.
- **No ve imágenes, solo texto.** Reacciona a la *descripción* de un anuncio, no
  al anuncio (rostro, color, jerarquía visual, tipografía) → hereda el sesgo de
  quien lo describe. Sí evalúa bien lo verbal (claims, mensajes, conceptos,
  naming, precio); no evalúa el impacto visual de una pieza gráfica o un video.
- Nota de internet ≠ conducta de compra; el arnés mide lo primero, que es una
  aproximación, no lo segundo.
