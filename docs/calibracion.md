# Calibración y backtest (Fase 3)

Cómo probamos que el enjambre **predice** y no solo suena bien. Basado en los
documentos de investigación de calibración (foco Chile → LATAM) y su paquete de
prompt de backtest.

## La idea

Un **backtest ciego**: al enjambre se le muestra la descripción de un ítem (una
app, un producto) **ocultándole el resultado real** (su nota promedio). El
enjambre predice; comparamos contra la realidad. Si acierta en casos que nunca
vio, tenemos evidencia de que funciona.

## Reglas de diseño (no negociables)

1. **Cegado.** El agente nunca ve la nota real ni el nº de reseñas. → `Item.estimulo()`.
2. **Sin números.** El agente reacciona en **texto libre**; la nota se calcula
   después por similitud semántica (SSR). Pedir un número aplasta la respuesta
   hacia el centro. → `PROMPT_SISTEMA_CHILE`, `ELICITACION_CHILE`, `SSR`.
3. **Muestreo estratificado.** Se eligen ítems repartidos por todo el rango de
   nota (malos, medios, buenos, excelentes) para **fabricar dispersión**. Sin
   esto, un modelo que siempre diga "4.3" parece genial. → `seleccion_estratificada`.
4. **Doble métrica.** Correlación **y** forma de la distribución. → `scoring`.
5. **Anti-autoengaño.** Baseline tonto (predecir la media), chequeo de dispersión
   de la muestra y de variedad de los agentes. → `reporte`.

## Umbrales de correlación

| r | Veredicto |
|---|-----------|
| ≥ 0.85 | **Excelente** (nivel *Nature*/SSR) — objetivo declarado |
| 0.70–0.85 | Bueno (screening/tamizaje) |
| 0.50–0.70 | Aceptable (solo exploratorio) |
| < 0.50 o signo invertido | No fiable |

**Regla de oro:** si un backtest da r > 0.90 sin esfuerzo, **sospecha antes de
celebrar** — casi siempre es distribución colapsada, fuga de datos (el modelo ya
vio los casos), reseñas falsas en la fuente, o muestra sin dispersión.

## Datos demográficos (Chile)

`data/segmentos_chile.example.json` — Censo 2024 INE (género, edad, región) +
modelo GSE AIM 2023.

> **Advertencia GSE (bloqueante).** Los grupos **C3, D y E no están verificados**
> en fuente primaria y por eso **se excluyen**. El backtest corre solo sobre
> AB→C2 y así se declara — esto **sesga la audiencia hacia NSE alto/medio**. Para
> completar: descargar el "Manual GSE AIM 2023" (aimchile.cl/gse-chile) y llenar
> la tabla. **No inventar esos porcentajes.**

## Cómo correrlo

```bash
python examples/demo_fase3.py                          # LLM simulado (offline)
ANTHROPIC_API_KEY=... python examples/demo_fase3.py    # con Claude real
```

Con el **LLM simulado la correlación es ~0 a propósito**: el mock no lee el
estímulo, así que no puede predecir. Eso confirma que el arnés no hace trampa —
la señal solo aparece con un LLM real que razona sobre el texto.

## Límites honestos (declarados)

- **Nota de internet ≠ conducta de compra.** Predecir reseñas replica una
  encuesta contaminada (sesgo en J, reseñas falsas); no es lo mismo que predecir
  comportamiento real. Es la crítica más dura al sector.
- **40–60 casos** son prueba de concepto, no afirmación estadística fuerte.
- **Demografía ≠ persona completa.** Los mejores estudios usan perfiles ricos,
  no solo edad/NSE/región.
- **Fuga de datos.** Priorizar ítems recientes/poco conocidos y reportar aparte
  el desempeño en ítems que casi seguro no estaban en el entrenamiento.

## Datasets y legal (resumen)

- Los corpus grandes de reseñas en español (MELISA, MARC, TASS) son de licencia
  **"solo investigación"** → probablemente **no** usables en un producto de pago.
- **Ley 21.719** (datos personales, Chile) entra en vigor el **1-dic-2026**;
  diseñar "cumpliendo desde el inicio". Una reseña con seudónimo probablemente
  **es** dato personal. Ingerir datos de clientes para entrenar IA es un cambio
  de finalidad → consultar abogado.

## Competencia / vara de medir

Ninguna plataforma comercial (Aaru, Synthetic Users, Qualtrics…) tiene
publicación revisada por pares; el estándar del sector es un white paper con una
cifra auto-reportada. **ThinkNow** (sintético latino anclado a panel real) es el
benchmark más directo en LATAM. La vara honesta no es "¿superas 0.85?" sino
"¿predices conducta real con un holdout independiente, sin eco de datos de
entrenamiento?". Responder bien esa pregunta es la oportunidad de Rubicon Lab.

## Fuentes

El detalle completo (con marcas [VERIFICADO]/[INFERIDO]/[VACÍO], tablas
demográficas y enlaces) está en los documentos de investigación que originaron
esta fase: *"Set de calibración retrospectiva para audiencias sintéticas"* y
*"Prompt del enjambre — backtest de calibración"*. Referencias clave: INE Censo
2024, AIM GSE Chile, Ashokkumar/Hewitt et al. (*Nature* 2026), SSR (arXiv
2510.08338), Park et al. (arXiv 2411.10109).
