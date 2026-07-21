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

### ✅ Fase 1 — Personas ancladas en datos reales *(completa)*
Convertir el generador de personas de "muestreo por defecto" a personas
calibradas contra una fuente demográfica LATAM real y enriquecidas con RAG.
- **Entregables:** ✓ carga de segmentación desde JSON del cliente
  (`Segmentacion.desde_json`); ✓ distribuciones condicionales (edad|país);
  ✓ priors OCEAN por segmento; ✓ intereses por NSE; ✓ capa RAG
  (`conocimiento.py`, recuperador TF-IDF) que aterriza cada persona en
  reseñas/tickets reales; ✓ validación de distribución (`validacion.py`).
- **Éxito:** ✓ la audiencia sintética reproduce el objetivo (desv_max ≤ 0.05 en
  todas las marginales, n=2000); ✓ las personas citan contexto real en su
  system prompt. Ver `examples/demo_fase1.py`.

### ✅ Fase 2 — Elicitación calibrada (SSR) + LLM real *(completa)*
Reemplazar el `ExtractorRating` heurístico por el método **SSR** (Semantic
Similarity Rating) y enchufar Claude como LLM de producción.
- **Entregables:** ✓ `SSR` funcional (distribución sobre anclas Likert por
  similitud semántica + intención esperada continua); ✓ interfaz `Embedder`
  con `EmbedderLexico` offline para dev/tests y ranura para embeddings
  semánticos reales; ✓ elicitación intercambiable en el orquestador
  (ExtractorRating ↔ SSR); ✓ cliente `AnthropicClient` conectable con solo
  definir `ANTHROPIC_API_KEY`. Ver `examples/demo_fase2.py`.
- **Éxito:** ✓ SSR es monótono (texto negativo→positivo produce 1.05→4.94) y su
  intención es continua. *Pendiente para producción:* medir correlación contra
  un set histórico real con un embedder semántico (meta r ≥ 0.7 zero-shot).

### ✅ Fase 3 — Motor de calibración y backtesting *(completa)*
El foso operativo: medir sistemáticamente qué tan bien predice el enjambre.
- **Entregables:** ✓ arnés de backtest ciego (`backtest.py`) con persona chilena
  muestreada de datos reales (Censo 2024 + GSE AIM, GSE parcial declarado);
  ✓ elicitación por texto libre + anclas SSR en español de Chile (promedio de
  varios juegos); ✓ selección estratificada por rango de nota; ✓ métricas
  (`metricas.py`): correlación, forma de la distribución (KS), MAE, baseline
  tonto; ✓ controles anti-autoengaño (dispersión de muestra, variedad de
  agentes). Ver `docs/calibracion.md` y `examples/demo_fase3.py`.
- **Éxito:** ✓ el arnés corre end-to-end y reporta la doble métrica; con LLM
  simulado da correlación ~0 y los controles lo detectan (no supera al baseline)
  — prueba de que no hace trampa.
- **Cierre de calibración real:** ✓ embedders semánticos reales enchufables
  (`embeddings.py`: SentenceTransformers / OpenAI / Voyage) con fábrica
  `crear_embedder` (por nombre o `ENJAMBRE_EMBEDDER`); ✓ SSR cachea las anclas
  (no gasta llamadas por agente); ✓ runner reproducible `calibrar.py` que
  persiste reporte JSON con metadata (semilla, versión de anclas, embedder,
  modelo, hash del dataset) y registra en la DB del foso; ✓ **autoverificación
  offline** (`demo_calibracion_real.py`): un lector heurístico que SÍ lee da
  r≈0.72 y supera al baseline, mientras el mock ciego da ~0 — prueba de que la
  máquina surface señal real cuando el lector comprende.
- **Para el número real** (solo requiere recursos externos): `ANTHROPIC_API_KEY`
  + un embedder semántico (`pip install sentence-transformers`) + un dataset con
  nota real. Comando: `python -m enjambre.calibrar --embedder st --items <tu.json>`.

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
- [x] Fase 1 — personas ancladas (demografía real + condicionales + OCEAN + RAG + validación)
- [x] Fase 2 — SSR + LLM real (elicitación semántica intercambiable + Claude conectable)
- [x] Fase 3 — calibración/backtesting (arnés ciego Chile + doble métrica + anti-autoengaño)
- [ ] Fase 4 — interfaz
- [ ] Fase 5 — piloto
- [ ] Fase 6 — empaquetado y siguientes rubros
