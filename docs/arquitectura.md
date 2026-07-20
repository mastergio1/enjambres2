# Arquitectura · Enjambre 2

## Principio

El enjambre es un **motor de simulación de población**: convierte un estímulo en
la reacción agregada de una audiencia sintética. La misma primitiva que el
Enjambre 1 de finanzas (una población de agentes que delibera), con dos cambios:
el **rol** de los agentes (consumidores en vez de analistas) y los **datos de
anclaje** (demografía LATAM en vez de mercados financieros).

## Flujo

```
        estímulo (anuncio / concepto / mensaje)
                     │
                     ▼
        ┌─────────────────────────┐
        │   GeneradorPersonas      │  ← Segmentación LATAM (país, edad, NSE,
        │   demografía + OCEAN      │     urbanidad) + rasgos psicográficos
        │   (+ RAG en Fase 1)       │     + datos del cliente
        └───────────┬─────────────┘
                     │  N personas
                     ▼
        ┌─────────────────────────┐
        │        Enjambre          │  ← cada persona reacciona vía LLM
        │   (orquestador)          │     (system prompt = perfil de la persona)
        └───────────┬─────────────┘
                     │  reacciones en texto libre
                     ▼
        ┌─────────────────────────┐
        │      Elicitación         │  ← texto → intención de compra 1-5
        │   ExtractorRating / SSR  │     (SSR evita el sesgo del Likert directo)
        └───────────┬─────────────┘
                     │  distribución de intención
                     ▼
        ┌─────────────────────────┐
        │   Resultado / Ranking    │  → media, dispersión, top-2-box, A/B
        └───────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   RegistroCalibracion    │  ← guarda predicción; al llegar el
        │   predicción vs. real    │     resultado real, mide correlación
        └─────────────────────────┘
```

## Componentes

- **`personas.py`** — `GeneradorPersonas` muestrea de una `Segmentacion`
  (distribuciones editables por cliente). Cada `Persona` produce su propio
  `system_prompt`. Reproducible vía semilla.
- **`llm.py`** — `LLMClient` es la interfaz. `MockLLMClient` (determinista,
  para desarrollo) y `AnthropicClient` (Claude en producción) son
  intercambiables sin tocar el orquestador.
- **`elicitacion.py`** — `ExtractorRating` (MVP) lee el rating del texto;
  `SSR` (Fase 2) lo deriva por similitud semántica contra anclas Likert.
- **`enjambre.py`** — `Enjambre.reaccionar()` y `.comparar()` (ranking A/B/n).
  `Resultado` expone media, dispersión, distribución y top-2-box.
- **`calibracion.py`** — `RegistroCalibracion` en SQLite: registra
  predicciones, ingiere resultados reales y calcula correlación de Pearson.

## Decisiones

- **Interfaz de LLM desacoplada**: permite desarrollar y testear sin gastar
  tokens, y cambiar de proveedor o usar multi-modelo después.
- **Elicitación separada de la generación**: podemos mejorar el método de
  medición (heurístico → SSR → fine-tune) sin tocar el resto.
- **Calibración desde el día 1**: el log existe antes que la UI, porque el
  track record es el activo defendible.
- **Todo reproducible**: semillas fijas para poder comparar cambios de forma
  justa (clave para evaluar mejoras de calibración).
