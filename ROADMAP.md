# Roadmap · Enjambre 2 — Audiencias Sintéticas LATAM

> Segundo producto del portafolio de **Rubicon Lab**. Reutiliza la arquitectura
> de "enjambre de agentes" del Enjambre 1 (finanzas) y la aplica a un nuevo
> rubro: **testear productos, anuncios y mensajes con consumidores sintéticos
> antes de lanzarlos al mercado.**

Priorización de rubros: ver el catálogo completo (28 ideas rankeadas por moat +
reutilización). Este roadmap desarrolla el **#1**, que en la práctica fusiona la
idea #1 (localización LATAM) con la #3 (pre-test de anuncios): el mismo motor,
con el foso del mercado hispanohablante que ningún competidor ataca hoy.

## Por qué esta idea primero

- **Máxima reutilización** de tu enjambre existente: pasar de agentes-analistas
  a agentes-consumidores es un cambio de rol y de datos, no de arquitectura.
- **Mercado validado**: Aaru ($1.000M val.), Simile ($100M), Synthetic Users,
  Artificial Societies. La técnica tiene respaldo académico (Stanford/*Nature*
  2026, r≈0.85–0.90; método SSR de PyMC Labs ≈90% correlación, open source).
- **Foso propio**: calibración honesta + mercado LATAM/español desatendido.

## El foso (lo que nos hace defendibles)

1. **Personas ancladas en datos**, no inventadas (distribución demográfica real
   + OCEAN + RAG con datos del cliente).
2. **Calibración continua**: cada predicción se guarda y se compara contra el
   resultado real. Vendemos track record, no magia.
3. **Posicionamiento honesto**: early-warning / red-team / ranking de variantes,
   no oráculo. (La correlación cae a ~0.3 en productos genuinamente novedosos.)

---

## Fases

Estimaciones en semanas de trabajo enfocado. Cada fase cierra con un criterio
de éxito verificable.

### ✅ Fase 0 — Fundaciones *(en curso)*
Estructura del repo, arquitectura, y un esqueleto que corre end-to-end con un
LLM simulado (sin llaves de API).
- **Entregables:** paquete `enjambre` (personas, elicitación, orquestador,
  calibración), `examples/demo.py`, docs.
- **Éxito:** `python examples/demo.py` corre un A/B/C y rankea variantes. ✓

### Fase 1 — Personas ancladas en datos reales *(~1–2 sem)*
Convertir el generador de personas de "muestreo por defecto" a personas
calibradas contra una fuente demográfica LATAM real y enriquecidas con RAG.
- **Entregables:** carga de distribuciones por país/segmento desde datos del
  cliente (CSV/JSON); capa RAG opcional (reviews, tickets, social listening);
  perfiles psicográficos por segmento.
- **Éxito:** la distribución de la audiencia sintética reproduce la del panel
  objetivo dentro de un margen acordado; las personas citan contexto real.

### Fase 2 — Elicitación calibrada (SSR) + LLM real *(~2 sem)*
Reemplazar el `ExtractorRating` heurístico por el método **SSR** (Semantic
Similarity Rating) y enchufar Claude como LLM de producción.
- **Entregables:** `SSR` funcional con embeddings; cliente `AnthropicClient`
  productivo; control de costo (modo rápido/profundo); multi-modelo opcional.
- **Éxito:** en un set histórico de conceptos con intención de compra conocida,
  la correlación supera un umbral mínimo (meta: r ≥ 0.7 zero-shot).

### Fase 3 — Motor de calibración y backtesting *(~1–2 sem)*
El foso operativo: medir sistemáticamente qué tan bien predice el enjambre.
- **Entregables:** ampliar `RegistroCalibracion` con métricas por vertical,
  reportes de correlación, e ingestión de resultados reales; motor
  anti-homogeneidad (validar que la *varianza* sintética se parezca a la humana).
- **Éxito:** dashboard de calibración con ≥20 casos cerrados y correlación
  reportada honestamente por categoría.

### Fase 4 — Interfaz "sube y reacciona" *(~2–3 sem)*
Que un no-técnico suba un creativo/mensaje y reciba la reacción + ranking.
- **Entregables:** API (FastAPI) + UI web simple; export de reporte; soporte de
  variantes A/B/n; segmentación configurable desde la UI.
- **Éxito:** un usuario externo completa el flujo (subir → resultado) sin ayuda.

### Fase 5 — Piloto con primer cliente LATAM *(~3–4 sem)*
Validación real con un cliente y un dataset de calibración propio.
- **Entregables:** onboarding de un cliente, calibración con sus datos, informe
  de precisión, iteración.
- **Éxito:** el cliente confirma que la lectura del enjambre le ahorró tiempo/
  dinero frente a su research tradicional en al menos un caso.

### Fase 6 — Empaquetado y siguientes rubros
Convertir el motor en base reutilizable para los siguientes ideas del ranking
(#2 wind-tunnel corporativo, #8 war-gaming, etc.), que comparten la primitiva
de "población que delibera".

---

## Stack (por defecto, ajustable)

| Capa | Elección MVP | Notas |
|------|--------------|-------|
| Lenguaje | Python 3.10+ | Ecosistema de agentes/synthetic users |
| Orquestación | Propia (ligera) | Extender el orquestador del Enjambre 1; LangGraph si hace falta grafo de estado |
| LLM | Claude (`AnthropicClient`) | Cliente intercambiable; multi-modelo opcional |
| Elicitación | Método SSR (PyMC Labs) | Open source, ~90% correlación zero-shot |
| Personas | Demografía + OCEAN + RAG | Vector DB (pgvector/Qdrant) en Fase 1+ |
| Persistencia | SQLite → Postgres | Log de calibración desde el día 1 |
| Interfaz | CLI → FastAPI + web | UI en Fase 4 |

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| "Persona collapse" / homogeneidad | Motor de diversidad + validar varianza (Fase 3) |
| Sesgo a la persona promedio | Anclaje demográfico explícito (Fase 1) |
| Baja fidelidad en productos novedosos (~0.3) | Posicionar como early-warning, no oráculo |
| Validación circular | Calibración retrospectiva contra histórico (Fase 3) |

## Estado actual

- [x] Fase 0 — fundaciones y demo end-to-end
- [ ] Fase 1 — personas ancladas
- [ ] Fase 2 — SSR + LLM real
- [ ] Fase 3 — calibración/backtesting
- [ ] Fase 4 — interfaz
- [ ] Fase 5 — piloto
- [ ] Fase 6 — empaquetado y siguientes rubros
