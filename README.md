# Asignación Táctica de ETFs

Proyecto ETF2. Responde una pregunta de cartera, no de un activo suelto: **dado un capital repartido
entre SPY, QQQ, IWM, GLD y TLT, ¿cuánto debería tener en cada uno según mi perfil de riesgo y, cada mes,
qué conviene comprar, mantener o vender?** Cada día de mercado un proceso automático descarga los precios, los guarda en una
base de datos real y produce tres reportes:

| Reporte | Pregunta | Cómo se responde |
|---|---|---|
| **Descriptivo** | ¿Qué pasó? | Velas, retornos, volatilidad, correlación, RSI, MACD, ATR y distribución de retornos |
| **Predictivo** | ¿Qué puede pasar? | Un modelo de machine learning entrenado aquí estima la probabilidad de que cada ETF suba en el próximo mes, con validación *walk-forward* |
| **Prescriptivo** | ¿Compro, mantengo o vendo cada ETF este mes? | Pesos, decisión y nivel de alerta por ETF para tres perfiles de riesgo, y el tamaño de cada posición en dinero. Los números salen de reglas; un modelo de lenguaje (API de Gemini) redacta la explicación y el código la valida |

> Proyecto académico. No es asesoría de inversión ni ejecuta operaciones con dinero real.

## Arquitectura

```
 GitHub Actions (cron: lun-vie 18:00 Perú · sábados recarga completa)  ◄── botón «Actualizar ahora»
        │  dispara                                                         del panel (API de GitHub)
        ▼
 run_pipeline.py ── 1. descarga precios ──────────►  Yahoo Finance (yfinance)
        │           2. guarda (upsert)   ─────────►  ┐
        │           3. lee el histórico  ◄─────────  ├─ API REST de Supabase (HTTP)
        │           4. indicadores, pesos y backtest │
        │           5. modelo predictivo (walk-forward)
        │           6. recomendación por perfil ──►  API de Gemini (solo redacta; el código valida)
        │           7. guarda resultados ─────────►  ┘
        ▼
 Supabase · PostgreSQL   (precios, indicadores, pesos, backtest, predicciones, recomendaciones, bitácoras)
        ▲
        │  lectura con la clave pública (solo puede leer)
 Panel (Streamlit + Plotly, solo lectura)
```

Los cálculos se hacen sobre lo que está **guardado en la base** (no sobre lo recién descargado):
la base es la fuente de verdad, y repetir una corrida no duplica nada porque todas las escrituras
son *upserts*.

## La regla de pesos (explicable, sin caja negra)

Todos los parámetros están en `etf2/config.py` y se fijaron **antes** de mirar resultados.

1. **Peso base por riesgo.** Cada ETF recibe un peso proporcional a `1 / volatilidad` (60 días,
   anualizada): los más tranquilos pesan más (versión sencilla de paridad de riesgo).
2. **Inclinación por momentum.** El momentum es el promedio de los retornos a 3, 6 y 12 meses. Se
   compara entre los 5 ETFs (z-score) y mueve el peso entre ×0,5 y ×1,5.
3. **Filtro de momentum absoluto.** Si el retorno a 6 meses de un ETF es ≤ 0, su peso se reduce a la
   mitad.
4. **Límites.** Ningún ETF por debajo de 5 % ni por encima de 40 %. Los pesos se escalan hasta que,
   ya recortados, suman exactamente 1.

Cada fila de `pesos_sugeridos` trae un campo `motivo` con la explicación en lenguaje humano.

### Backtest

- Se decide al cierre del último día hábil de cada mes y se **ejecuta al cierre del día hábil
  siguiente** (un día de rezago, para no operar con un precio que recién se conoce).
- Costo de 0,1 % por cada unidad de capital negociada, incluida la compra inicial.
- Mismas fechas y mismos costos para todas las estrategias: «Asignación táctica» (la regla),
  «Pesos iguales» y «Solo SPY».
- Métricas: retorno total, CAGR, volatilidad, Sharpe (tasa libre de riesgo = 0), máximo drawdown y
  rotación anual (*turnover*).

## Reporte predictivo (`etf2/predictivo.py`)

**Qué predice.** Para cada ETF y cada día: ¿el precio ajustado estará más alto dentro de 21 días
hábiles (≈ un mes)? Es una clasificación binaria; la salida es una probabilidad.

**Con qué.** Diez variables calculadas solo con precios hasta ese día: retornos a 1, 3, 6 y 12 meses,
volatilidad a 60 días, distancia al máximo de 12 meses, RSI (14), histograma del MACD sobre el precio,
distancia a la media de 50 días y volatilidad de 20 días sobre la de 60.

**Modelos.** Regresión logística (con las variables estandarizadas; es el modelo principal) y
*gradient boosting* de árboles poco profundos, ambos de scikit-learn y con hiperparámetros fijos.
Se comparan con dos referencias que no aprenden nada: «sube si su retorno a 6 meses es positivo» y
«sube con la frecuencia histórica».

**Cómo se evita mirar el futuro.**

- *Walk-forward*: cada fin de mes se entrena un modelo nuevo con todo lo anterior y predice el mes
  siguiente. Todas las métricas son fuera de muestra.
- *Embargo*: el resultado de un caso (si subió en los 21 días siguientes) solo se conoce 21 días
  después, así que se entrena únicamente con casos cuyo resultado ya estaba cerrado el día de la
  predicción.
- El escalado se ajusta dentro de cada entrenamiento.
- Hay pruebas automáticas: si se alteran todos los precios posteriores a una fecha, la predicción de
  esa fecha no cambia.

**Cómo se evalúa.** Exactitud, precisión, sensibilidad, F1, AUC, Brier e IC de Spearman, más una
prueba de ablación: la misma regla de pesos **con y sin** la inclinación del modelo (máximo ±30 % por
ETF), en el mismo periodo y con los mismos costos. De ahí salen el retorno simulado con comisiones y
el máximo drawdown frente a las referencias.

**Honestidad.** Predecir el mercado a un mes es difícil: lo normal es un AUC apenas sobre 0,50. El
panel dice sin adornos si el modelo le gana o no a la referencia simple, y **la recomendación solo usa
el modelo si mejoró el Sharpe de la regla fuera de muestra**. Si no, se usa la regla sola.

## Reporte prescriptivo (`etf2/prescriptivo.py`)

**Perfiles.** Cada perfil fija el rango de renta variable (SPY + QQQ + IWM) de la cartera:
Conservador 20–45 %, Moderado 45–65 %, Agresivo 65–85 %. Si los pesos de partida quedan fuera del
rango, la renta variable se lleva al borde más cercano, sin romper el mínimo de 5 % ni el tope de 40 %
por ETF.

**La decisión del mes: ¿compro, mantengo o vendo cada ETF?** Se compara el peso del perfil con el del
cierre del mes anterior:

- **Comprar:** el peso sube 2 puntos o más (comprar más de ese ETF).
- **Vender:** el peso baja 2 puntos o más (vender una parte; ningún ETF baja de 5 %, así que nunca se
  vende todo).
- **Mantener:** el peso cambia menos de 2 puntos.

**Tamaño de la posición.** La calculadora convierte los pesos del perfil en dinero por ETF para el
capital que se indique, y en las compras y ventas necesarias si ya se tiene una cartera.

**Nivel de alerta.** Para cada ETF: `último cierre × (1 − 2 × volatilidad anual / √12)`, es decir, el
precio tras una caída de dos veces la volatilidad de un mes. Si el ETF cierra por debajo, conviene
revisar la posición antes del próximo rebalanceo. **No es un stop loss:** la simulación histórica no
vende cuando se cruza ese nivel, así que no se presenta como una regla probada.

**Reparto del trabajo.**

| Lo decide el código (reglas) | Lo redacta el LLM |
|---|---|
| Pesos de cada perfil | Resumen de la cartera |
| Cambio frente al cierre del mes anterior | Explicación de cada ETF |
| Decisión por ETF: Comprar / Vender (cambio ≥ 2 puntos) o Mantener | Riesgos |
| Nivel de alerta de cada ETF | |
| Si se usa o no el modelo predictivo, y su nivel de confianza | Qué vigilar |

El LLM recibe un JSON con esos números ya calculados y devuelve otro JSON. No puede cambiar pesos ni
acciones.

**Revisiones antes de guardar el texto** (todas por código):

1. Formato: JSON con los campos pedidos y los cinco ETFs, una vez cada uno.
2. Acciones: la acción de cada ETF coincide con la calculada, y el resumen nombra los ETFs que se
   compran y los que se venden. La frase sobre si se usa el modelo la escribe el código, no el LLM.
3. Cifras: todo número del texto existe en los datos enviados (se aceptan redondeos).
4. Lenguaje: sin promesas de resultado, precios objetivo, «vender todo» ni «se debe comprar o vender».

Si el texto falla, se pide una corrección indicando el motivo; si vuelve a fallar, o la API da error,
o no hay clave, o se alcanzó el límite, la explicación sale de una **plantilla fija** con los mismos
datos. El panel muestra quién redactó cada recomendación.

**Salvaguardas.** Máximo 12 llamadas por día y 6 por ejecución; tope de tokens de respuesta y tiempo
de espera; la clave viaja en una cabecera HTTP y solo existe en `.env` o en GitHub Secrets; si los
datos no cambiaron no se vuelve a llamar (se compara una huella de la entrada); cada llamada queda en
la tabla `llm_log` con estado, tokens y duración. El panel solo lee: quien lo visita no puede
provocar llamadas al LLM.

**Proveedor.** Google Gemini por HTTP directo (`generateContent`), sin SDK. El modelo se elige de
`LLM_MODELOS` en `etf2/config.py` (si el primero ya no existe se usa el siguiente) o con la variable
`GEMINI_MODEL`.

## Base de datos

| Tabla | Contenido |
|---|---|
| `etfs` | Catálogo de los 5 ETFs |
| `precios_diarios` | Apertura, máximo, mínimo, cierre, cierre ajustado y volumen por ETF y día |
| `indicadores` | Retornos 1/3/6/12 meses, volatilidad 60d, momentum y drawdown |
| `pesos_sugeridos` | Peso final, peso base, factor de momentum y motivo, por fecha y ETF |
| `backtest_resumen`, `backtest_curva` | Métricas y curvas de capital de la regla y sus referencias |
| `modelo_predicciones` | Probabilidad de subida por fecha, ETF y modelo, y qué pasó después |
| `modelo_metricas`, `modelo_importancias` | Desempeño fuera de muestra y peso de cada variable |
| `pesos_ml` | La regla con la inclinación del modelo |
| `backtest_ml_resumen`, `backtest_ml_curva` | La regla con y sin modelo, mismo periodo |
| `recomendaciones` | Una fila por fecha y perfil: quién redactó, resumen y detalle (JSON) |
| `llm_log` | Cada llamada al LLM: estado, tokens, duración y motivo de rechazo |
| `corridas` | Bitácora: cada ejecución con su fecha, estado, origen y filas procesadas. Origen: `programada` (horario de GitHub), `panel` (botón «Actualizar ahora»), `github_manual` («Run workflow») o `pc` (desde una computadora) |
| `v_pesos_actuales` | Vista con los pesos de la última fecha |

Seguridad en dos capas: `anon` solo recibe permiso de lectura, y además hay políticas de seguridad
por fila (RLS). Solo la clave secreta, guardada en GitHub Secrets, puede escribir.

## Puesta en marcha

### 1. Supabase

1. Crea una cuenta y un proyecto nuevo (por ejemplo `ETF2`).
2. Abre **SQL Editor → New query**, pega todo `sql/schema.sql` y pulsa **Run**.
   Si la base ya existía con la versión anterior, ejecuta solo `sql/migracion_ml_llm.sql`: agrega
   las tablas de los reportes predictivo y prescriptivo sin tocar los datos.
3. En **Settings → API Keys** copia la clave **secret** (`sb_secret_...`); en **Connect** (o
   Settings → API) copia la **Project URL**. La clave *publishable* es la que usa el panel.

### 2. Clave de Gemini (opcional)

En Google AI Studio crea una clave de API. Va en `.env` (y luego en GitHub Secrets) como
`GEMINI_API_KEY`. Sin ella todo funciona igual, con la explicación por plantilla.

### 3. GitHub

1. Crea un repositorio **nuevo** (`ETF2`) y sube esta carpeta (el `.env` no se sube: está en
   `.gitignore`).
2. En **Settings → Secrets and variables → Actions → New repository secret** crea
   `SUPABASE_URL`, `SUPABASE_SECRET_KEY` y, si la tienes, `GEMINI_API_KEY`.
3. En **Actions → "ETF2 · actualización diaria" → Run workflow**, elige el modo `historico`.
   Desde entonces corre solo de lunes a viernes a las 18:00 (hora de Perú), y los sábados hace una
   recarga completa para recoger los ajustes de Yahoo por dividendos.

### 4. Botón «Actualizar ahora» (opcional)

El panel puede pedirle a GitHub que corra el workflow en el momento, sin esperar a las 18:00. El panel
no ejecuta el proceso ni escribe en la base: solo envía la orden (`workflow_dispatch`) y muestra el
avance; el trabajo lo hace GitHub Actions con sus propios secretos.

1. En GitHub: foto de perfil → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**. En *Repository access* elige **Only select
   repositories** y marca solo `ETF2`; en *Permissions → Repository permissions* pon **Actions: Read
   and write**. Copia el token (empieza con `github_pat_`).
2. En los *Secrets* del panel (Streamlit Cloud) o en tu `.env`, agrega `PANEL_GITHUB_TOKEN`,
   `PANEL_GITHUB_REPO` (`tu-usuario/ETF2`) y `PANEL_CLAVE` (la contraseña que pedirá el botón).
3. El botón aparece en la **cabecera**, junto a «Actualizado hace …», en todas las pestañas. Abre una
   ventana con el tipo de carga y la contraseña, no lanza nada si ya hay una ejecución
   en curso y, cuando GitHub termina, el panel vuelve a leer la base solo.

Sin esas tres variables el botón no aparece y todo lo demás funciona igual.

Dos avisos: GitHub puede desactivar los workflows programados de un repositorio sin actividad
durante mucho tiempo, y los proyectos del plan gratuito de Supabase pueden pausarse por inactividad.

### Variables de entorno

| Variable | Quién la usa | Para qué |
|---|---|---|
| `SUPABASE_URL` | pipeline y panel | Dirección del proyecto |
| `SUPABASE_SECRET_KEY` | solo el pipeline | Escribir en la base |
| `SUPABASE_PUBLISHABLE_KEY` | solo el panel | Leer de la base |
| `GEMINI_API_KEY` | solo el pipeline (opcional) | Redactar las recomendaciones |
| `GEMINI_MODEL` | solo el pipeline (opcional) | Elegir otro modelo de Gemini |
| `PANEL_GITHUB_TOKEN` | solo el panel (opcional) | Lanzar el workflow desde el botón |
| `PANEL_GITHUB_REPO` | solo el panel (opcional) | Repositorio del workflow (`usuario/ETF2`) |
| `PANEL_CLAVE` | solo el panel (opcional) | Contraseña del botón «Actualizar ahora» |

### Probar en tu computadora

```bash
pip install -r requirements.txt

# Sin base de datos: calcula y muestra pesos, backtest, métricas del modelo y recomendaciones
python run_pipeline.py --modo historico --sin-db

# Con base de datos: copia .env.example como .env, rellénalo y ejecuta
python run_pipeline.py --modo historico

# El panel
streamlit run dashboard.py
```

Opciones: `--sin-ml` omite el modelo predictivo; `--sin-llm` no llama al modelo de lenguaje;
`--csv ruta/precios.csv` lee los precios de un CSV (`Date,Open,High,Low,Close,Volume,Ticker`).
La carga histórica entrena un modelo por cada mes desde 2018, así que tarda un par de minutos.

### Publicar el panel

- **Streamlit Community Cloud:** despliega desde el repositorio de GitHub. En *Secrets* van solo
  `SUPABASE_URL` y `SUPABASE_PUBLISHABLE_KEY` (ver `.streamlit/secrets.toml.example`). La clave
  secreta y la de Gemini nunca van ahí: el panel no las necesita y rechaza una clave secreta. Si
  quieres el botón «Actualizar ahora», agrega también las tres variables `PANEL_…` (paso 4).
- **Docker:** `docker build -t etf2 .` y `docker run -p 8501:8501 --env-file .env etf2`.

## El panel

| Pestaña | Qué muestra |
|---|---|
| **Resumen** | La decisión del mes (¿compro, mantengo o vendo cada ETF?) con su nivel de alerta, el reparto de la regla, la lectura del día, los tres reportes en una línea cada uno, la calculadora del tamaño de la posición y la curva del backtest |
| **Descriptivo** | Panorama de los cinco ETFs, velas japonesas con medias móviles y volumen, RSI, MACD y ATR, rendimiento comparado, correlaciones y distribución de retornos diarios. Filtros por ETF y por rango |
| **Predictivo** | Probabilidad de subida a un mes por ETF, métricas fuera de muestra frente a las referencias, calibración, matriz de confusión, peso de cada variable y la prueba «con y sin modelo» |
| **Prescriptivo** | Selector de perfil de riesgo, recomendación con su explicación, decisión y nivel de alerta por ETF, el porqué de cada ETF, riesgos, qué vigilar, tamaño de la posición para el perfil, cómo se generó y la bitácora del LLM con su costo estimado |
| **Asignación** | Del peso por riesgo al peso final, peso frente a aporte al riesgo, señales de hoy y evolución mensual |
| **Backtest** | Crecimiento de 100 invertidos, métricas, caídas desde el máximo, retorno por año y mapa mensual |
| **Datos y método** | Arquitectura, cobertura de los datos, avance de las ejecuciones en GitHub, bitácora de ejecuciones, método, límites y fases CRISP-DM |

- **Nombre del panel:** se cambia en `etf2/config.py` (`NOMBRE_PANEL`, `ROTULO_PANEL` y `LEMA_PANEL`).
- **«Actualizado hace …»:** la cabecera dice cuándo terminó la última corrida correcta y cómo arrancó:
  automático, botón del panel, manual en GitHub o desde una PC.
- **Colores:** están en `etf2/tema.py` y en `.streamlit/config.toml`. Cada ETF conserva su color en
  todos los gráficos; verde y rojo quedan reservados para subidas y bajadas.
- **Accesibilidad:** cada gráfico tiene debajo su tabla de datos, y las variaciones llevan siempre
  signo o flecha además del color.
- **Color con función:** el dorado se reserva para lo que se puede tocar y lo elegido; verde y rojo,
  para subidas, bajadas, comprar y vender. Los rótulos y textos de apoyo van en gris.
- **Interacción:** botones, selectores, pestañas y desplegables se aclaran y crecen un poco al pasar
  el mouse, y se hunden al hacer clic (sin animación si el sistema pide movimiento reducido). Los
  encabezados con línea punteada explican la métrica al pasar el mouse.
- **Movimiento:** el contenido entra con un fundido corto al cambiar de pestaña, los indicadores clave
  aparecen en orden, las barras de reparto se llenan de izquierda a derecha y el punto de «Datos al
  día» late. Los gráficos se vuelven a crear al cambiar un filtro, así que entran con un fundido. Todo
  se apaga si el sistema pide movimiento reducido.
- **Ícono:** `assets/icono.png` es el ícono de la pestaña del navegador.

### Solución de problemas

- **401 / 403:** revisa que la clave sea la del proyecto correcto.
- **404 en una tabla:** falta ejecutar `sql/schema.sql` (o `sql/migracion_ml_llm.sql` si el aviso
  habla de las tablas del modelo y las recomendaciones).
- **"Yahoo Finance no devolvió datos":** suele ser temporal; vuelve a lanzar el workflow.
- **El botón dice que el token no tiene permiso o que no encuentra el repositorio:** el token debe
  tener acceso a `ETF2` con **Actions: Read and write**, y `PANEL_GITHUB_REPO` debe ser
  `usuario/ETF2`.
- **Las recomendaciones dicen «Texto por plantilla»:** falta `GEMINI_API_KEY`, se alcanzó el límite
  diario o el texto fue rechazado; el motivo está en la tabla `llm_log` y en la pestaña Prescriptivo.

## Límites que conviene decir en voz alta

- El universo son 5 ETFs conocidos hoy: eso introduce un sesgo de selección.
- El resultado depende del periodo analizado; el pasado no garantiza nada.
- El modelo predictivo tiene pocas observaciones independientes (una decisión por mes y cinco ETFs
  que se mueven juntos); su desempeño real puede ser peor que el medido.
- No se modelan impuestos ni diferencias entre precio de compra y de venta más allá de la fricción.
- `yfinance` es una librería no oficial que usa datos de Yahoo Finance, pensada para uso educativo.
- Un modelo de lenguaje puede redactar mal; por eso no decide nada y su texto pasa por revisiones.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

184 pruebas, ninguna usa servicios externos (ni Yahoo, ni Supabase, ni Gemini, ni GitHub): indicadores sin
mirar al futuro; límites de pesos con miles de casos aleatorios; rezago y costos del backtest; el
modelo predictivo (embargo, predicción insensible a precios futuros, reproducibilidad, ablación);
perfiles de riesgo, decisión del mes y nivel de alerta; la revisión del texto del LLM (cifras inventadas, acciones cambiadas, frases
prohibidas, JSON roto), el reintento y la plantilla de respaldo; el cliente de Gemini con una sesión
HTTP de mentira (modelo retirado, saturación, clave rechazada); límites de llamadas; el pipeline
completo contra un servidor que imita a Supabase (idempotencia, base sin migrar, errores); el cliente
de GitHub del botón «Actualizar ahora» contra un servidor que imita su API (permisos, repositorio mal
escrito, sin conexión) y el botón en el panel (contraseña, ejecución en curso, aviso al terminar); los
cálculos y gráficos del panel; y el panel de punta a punta, incluida la comprobación de que nunca
escribe en la base, rechaza una clave secreta y escapa el texto que viene del LLM.

El esquema y la migración se probaron además en un PostgreSQL real, reproduciendo todas las
escrituras del pipeline y verificando que el rol público puede leer pero no escribir.

## Metodología CRISP-DM · dónde está cada fase

| Fase | Dónde |
|---|---|
| Comprensión del negocio | Este README (objetivo, perfiles, límites) |
| Comprensión de los datos | `etf2/datos.py` (validación), tabla `precios_diarios`, pestaña Descriptivo |
| Preparación de los datos | `etf2/indicadores.py`, `etf2/tecnico.py` |
| Modelado | `etf2/pesos.py` (regla) y `etf2/predictivo.py` (modelo) |
| Evaluación | `etf2/backtest.py`, validación walk-forward en `etf2/predictivo.py`, `tests/` |
| Despliegue | `.github/workflows/diario.yml`, Supabase, `etf2/prescriptivo.py`, `dashboard.py`, `Dockerfile` |

## Estructura

```
ETF2/
├─ run_pipeline.py          # punto de entrada del proceso de actualización
├─ dashboard.py             # el panel (solo lee de la base)
├─ assets/icono.png         # ícono de la pestaña del navegador
├─ etf2/
│  ├─ config.py             # ETFs, parámetros del método, perfiles y ajustes del LLM
│  ├─ actualizacion.py      # botón «Actualizar ahora»: lanza y sigue el workflow en GitHub
│  ├─ datos.py              # descarga y validación de precios
│  ├─ indicadores.py        # retornos, volatilidad, momentum, drawdown
│  ├─ tecnico.py            # RSI, MACD y ATR
│  ├─ pesos.py              # regla de pesos
│  ├─ backtest.py           # simulación con costos
│  ├─ predictivo.py         # modelo de ML con validación walk-forward
│  ├─ prescriptivo.py       # perfiles, acciones, LLM y revisión de su texto
│  ├─ api_db.py             # cliente de la API REST de Supabase
│  ├─ pipeline.py           # orquesta todo: extraer → guardar → calcular → predecir → recomendar
│  ├─ panel.py              # cálculos del panel
│  ├─ graficos.py           # gráficos de Plotly
│  ├─ componentes.py        # piezas de HTML (cabecera, tablas, fichas)
│  ├─ tema.py               # colores y hoja de estilos
│  └─ formato.py            # números y fechas en español
├─ sql/schema.sql           # tablas, permisos y seguridad de la base de datos
├─ sql/migracion_ml_llm.sql # solo las tablas nuevas, para una base que ya existía
├─ Dockerfile               # imagen del panel
├─ .env.example             # nombres de las variables (sin valores)
├─ .streamlit/config.toml   # tema del panel
├─ .github/workflows/       # disparador diario
└─ tests/
```
