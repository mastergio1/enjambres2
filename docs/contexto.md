# Contexto del proyecto · Enjambre 2

## Qué es

**Enjambre 2** es un motor de **audiencias sintéticas**: un enjambre de agentes
de IA que actúan como consumidores reales latinoamericanos. Le muestras un
producto, un anuncio o un mensaje, y el enjambre **reacciona como lo haría tu
público objetivo** — te dice qué opina, cuánto le interesa y qué variante
prefiere, en minutos y sin salir a la calle a encuestar.

Es el **segundo producto del portafolio de Rubicon Lab**. Reutiliza la
arquitectura de "enjambre de agentes" del Enjambre 1 (finanzas) y la lleva a un
nuevo rubro: **investigación de mercado y validación de productos**.

## Para qué sirve

Sirve para **reducir la incertidumbre y descartar ideas débiles antes de gastar
dinero real**. Hoy validar un producto o una campaña con investigación de
mercado tradicional cuesta miles de dólares y semanas de trabajo (focus groups,
encuestas, paneles). Enjambre 2 da una **primera lectura barata y rápida** para:

- Elegir entre varias versiones de un anuncio o mensaje (**A/B/n**).
- Estimar la **intención de compra** de un concepto antes de fabricarlo.
- Descubrir **qué segmentos** reaccionan mejor o peor.
- Detectar objeciones y riesgos ("esto suena caro", "no le veo el beneficio").
- Priorizar un **roadmap** de features según reacción por segmento.

> No reemplaza al cliente real: se usa como **early-warning, red-team y filtro
> de priorización**, no como oráculo. Las decisiones de alto riesgo se siguen
> validando con personas reales. (Ver "Límites y honestidad".)

## Para quién es

| Cliente | Qué gana |
|---------|----------|
| **PYMEs y emprendedores** | Validan ideas sin pagar research caro |
| **Agencias y equipos de marketing** | Pre-testean campañas y creatividades antes del gasto en medios |
| **Áreas de producto** | Priorizan features y mensajes por segmento |
| **Marcas de consumo (CPG)** | Prueban packaging, naming, claims y precios |

## Por qué LATAM

Casi todos los competidores (Aaru, Synthetic Users, Artificial Societies) están
enfocados en mercados anglosajones. Un enjambre afinado para el **consumidor
latino** —su cultura, su lenguaje, sus niveles socioeconómicos— es un hueco de
mercado sin dueño y el posicionamiento más defendible del producto.

## Casos de uso concretos

### 1. Pre-test de anuncios (el caso central)
Una marca de bebidas tiene 3 versiones de un post para lanzar un yogurt
proteico. En vez de adivinar o gastar en pauta para probar, corre el enjambre:
obtiene intención de compra y *top-2-box* por variante y por país, y lanza solo
la ganadora. **Ahorra el gasto en las creatividades que iban a fallar.**

### 2. Validación de concepto de producto
Un emprendedor quiere saber si su idea de "café en cápsula compostable" tiene
tracción antes de invertir en producción. El enjambre estima la intención de
compra y expone las objeciones más comunes por segmento.

### 3. Naming y pricing
Una startup evalúa 15 nombres de marca y 3 puntos de precio. El enjambre rankea
por memorabilidad/connotación y por disposición a pagar, filtrando las opciones
débiles antes del research humano final.

### 4. Testeo de mensajes y claims
Un equipo de marketing prueba si el ángulo "salud", "precio" o "sostenibilidad"
conecta mejor con cada nivel socioeconómico — y descubre, por ejemplo, que la
sostenibilidad mueve al NSE alto pero no al bajo.

### 5. Priorización de features de una app
Un product manager simula la reacción de sus segmentos de usuarios a 5 features
propuestas, y ordena el roadmap por impacto percibido antes de programar nada.

### 6. Localización cultural
Una marca que va a entrar a México, Colombia y Argentina prueba cómo cae el
mismo mensaje en cada mercado, ajustando tono y referencias por país.

### Rubros futuros (mismo motor)
La primitiva "población de agentes que delibera" habilita, más adelante, otros
productos del catálogo de Rubicon Lab: wind-tunnel regulatorio corporativo,
war-gaming empresarial y polling de opinión — reutilizando este mismo núcleo.

## Cómo funciona (resumen)

1. **Personas ancladas** — se generan consumidores sintéticos muestreados de una
   distribución demográfica LATAM real (país, edad, NSE, urbanidad), con rasgos
   de personalidad (OCEAN) e intereses por segmento.
2. **Aterrizaje con datos (RAG)** — cada persona reacciona teniendo a la vista lo
   que gente parecida dijo de verdad (reseñas, tickets del cliente).
3. **Reacción del enjambre** — cada persona opina en lenguaje natural sobre el
   estímulo.
4. **Medición (SSR)** — la reacción se convierte a intención de compra 1-5 por
   similitud semántica contra anclas, evitando el sesgo de pedir un número.
5. **Resultado** — media, distribución, *top-2-box* y ranking A/B/n.
6. **Calibración** — cada predicción se guarda y se compara contra el resultado
   real cuando llega; ese track record es el activo defendible.

Detalle técnico en [`arquitectura.md`](arquitectura.md); plan por fases en
[`../ROADMAP.md`](../ROADMAP.md).

## Límites y honestidad (importante para vender)

- **Fidelidad variable**: las simulaciones predicen bien a nivel agregado en
  categorías conocidas (correlaciones altas en la literatura), pero **caen en
  productos genuinamente novedosos** (correlaciones tan bajas como ~0.3).
- **Riesgo de homogeneidad**: los agentes tienden a parecerse entre sí más que
  los humanos reales; por eso anclamos en demografía real y validamos la
  varianza, no solo la media.
- **No sustituye la voz del cliente real** en decisiones de alto riesgo.

El posicionamiento honesto —"te decimos dónde funciona y dónde no, y lo
probamos contra datos reales"— es también el más defendible legal y
comercialmente. Vendemos **reducción de incertidumbre**, no certezas.

## Referentes del mercado

- **PyMC Labs / método SSR** — open source, ~90% de correlación con intención de
  compra real (caso Colgate).
- **Aaru** ($1.000M val.), **Simile** ($100M), **Synthetic Users**,
  **Artificial Societies** — el mercado ya está validado y financiado.
- **Stanford / *Nature* 2026** — los LLM predicen experimentos sociales con
  r≈0.85–0.90.
