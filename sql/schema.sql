-- =============================================================================
-- ETF2 · esquema de la base de datos (PostgreSQL / Supabase)
--
-- Cómo usarlo: en Supabase abre "SQL Editor" -> "New query", pega TODO este
-- archivo y pulsa "Run". Es seguro ejecutarlo más de una vez.
--
-- Diseño en una línea: el pipeline (con la clave service_role) ESCRIBE; el
-- dashboard (con la clave anon) solo puede LEER. Eso lo imponen las políticas
-- de seguridad por fila (RLS) del final de este archivo.
-- =============================================================================

-- Catálogo de ETFs del universo ------------------------------------------------
create table if not exists public.etfs (
    ticker        text primary key,
    nombre        text not null,
    clase_activo  text not null,
    descripcion   text
);

-- Bitácora de ejecuciones del pipeline (evidencia de que la automatización corre)
create table if not exists public.corridas (
    id                 bigint generated always as identity primary key,
    inicio             timestamptz not null default now(),
    fin                timestamptz,
    estado             text not null default 'en_curso'
                       check (estado in ('en_curso', 'ok', 'error')),
    modo               text not null check (modo in ('historico', 'diario')),
    origen             text not null default 'manual',   -- 'programada', 'panel', 'github_manual' o 'pc'
                                                         -- (antes del 18-nov: 'github_actions' o 'manual')
    filas_precios      integer,
    filas_indicadores  integer,
    filas_pesos        integer,
    mensaje            text
);

-- Precios diarios (una fila por ETF y día) ------------------------------------
create table if not exists public.precios_diarios (
    ticker           text not null references public.etfs (ticker),
    fecha            date not null,
    apertura         numeric(14, 4),
    maximo           numeric(14, 4),
    minimo           numeric(14, 4),
    cierre           numeric(14, 4) not null,
    cierre_ajustado  numeric(14, 6) not null,   -- incluye dividendos y splits
    volumen          bigint,
    primary key (ticker, fecha)
);
create index if not exists idx_precios_fecha on public.precios_diarios (fecha);

-- Indicadores calculados (una fila por ETF y día) -----------------------------
create table if not exists public.indicadores (
    ticker     text not null references public.etfs (ticker),
    fecha      date not null,
    ret_1m     numeric(12, 6),
    ret_3m     numeric(12, 6),
    ret_6m     numeric(12, 6),
    ret_12m    numeric(12, 6),
    vol_60d    numeric(12, 6),   -- volatilidad anualizada a 60 días
    momentum   numeric(12, 6),   -- promedio de ret_3m, ret_6m y ret_12m
    drawdown   numeric(12, 6),   -- distancia al máximo de 12 meses
    primary key (ticker, fecha)
);

-- Pesos sugeridos por la regla de ETF2 ----------------------------------------
create table if not exists public.pesos_sugeridos (
    fecha            date not null,
    ticker           text not null references public.etfs (ticker),
    peso             numeric(8, 6) not null check (peso >= 0 and peso <= 1),
    peso_base        numeric(8, 6),    -- peso solo por riesgo (1/volatilidad)
    factor_momentum  numeric(8, 4),    -- multiplicador total por momentum
    motivo           text,             -- explicación en lenguaje humano
    corrida_id       bigint references public.corridas (id) on delete set null,
    primary key (fecha, ticker)
);
create index if not exists idx_pesos_fecha on public.pesos_sugeridos (fecha desc);

-- Resultado del backtest (una fila por estrategia) ----------------------------
create table if not exists public.backtest_resumen (
    estrategia      text primary key,
    fecha_inicio    date,
    fecha_fin       date,
    retorno_total   numeric(12, 6),
    cagr            numeric(12, 6),
    volatilidad     numeric(12, 6),
    sharpe          numeric(12, 6),
    max_drawdown    numeric(12, 6),
    turnover_anual  numeric(12, 6),
    n_rebalanceos   integer,
    friccion        numeric(8, 6),
    actualizado_en  timestamptz default now()
);

-- Curvas de capital del backtest (base 1 en el primer día) --------------------
create table if not exists public.backtest_curva (
    fecha          date primary key,
    estrategia     numeric(14, 6),
    pesos_iguales  numeric(14, 6),
    spy            numeric(14, 6)
);

-- Modelo predictivo: probabilidad de que cada ETF suba en el próximo mes ----------
-- Una fila por fecha, ETF y modelo. Cada probabilidad se calculó con un modelo entrenado
-- solo con datos anteriores a su fecha (validación walk-forward).
create table if not exists public.modelo_predicciones (
    fecha       date not null,
    ticker      text not null references public.etfs (ticker),
    modelo      text not null,        -- logit, gb, momentum (referencia) o frecuencia (referencia)
    prob_sube   numeric(8, 6) not null check (prob_sube >= 0 and prob_sube <= 1),
    subio       smallint check (subio in (0, 1)),   -- qué pasó después; vacío si aún no se sabe
    ret_futuro  numeric(12, 6),
    primary key (fecha, ticker, modelo)
);
create index if not exists idx_predicciones_fecha on public.modelo_predicciones (fecha desc);

-- Desempeño fuera de muestra de cada modelo (una fila por modelo) ----------------
create table if not exists public.modelo_metricas (
    modelo          text primary key,
    n               integer,
    n_fechas        integer,
    fecha_desde     date,
    fecha_hasta     date,
    exactitud       numeric(8, 6),
    precision_pos   numeric(8, 6),
    sensibilidad    numeric(8, 6),
    f1              numeric(8, 6),
    auc             numeric(8, 6),
    brier           numeric(8, 6),
    ic_spearman     numeric(8, 6),
    tasa_positivos  numeric(8, 6),
    actualizado_en  timestamptz default now()
);

-- Peso de cada variable en el último modelo ajustado -----------------------------
create table if not exists public.modelo_importancias (
    modelo       text not null,
    variable     text not null,
    importancia  numeric(12, 6),
    signo        smallint,            -- solo la regresión logística: +1 empuja a «sube», -1 a «no sube»
    primary key (modelo, variable)
);

-- La misma regla de pesos con la inclinación del modelo (fines de mes y última fecha)
create table if not exists public.pesos_ml (
    fecha            date not null,
    ticker           text not null references public.etfs (ticker),
    peso             numeric(8, 6) not null check (peso >= 0 and peso <= 1),
    peso_base        numeric(8, 6),
    factor_momentum  numeric(8, 4),
    prob_sube        numeric(8, 6),
    factor_ml        numeric(8, 4),
    motivo           text,
    primary key (fecha, ticker)
);

-- Backtest de la regla con y sin el modelo, en el mismo periodo fuera de muestra ---
create table if not exists public.backtest_ml_resumen (
    estrategia      text primary key,
    fecha_inicio    date,
    fecha_fin       date,
    retorno_total   numeric(12, 6),
    cagr            numeric(12, 6),
    volatilidad     numeric(12, 6),
    sharpe          numeric(12, 6),
    max_drawdown    numeric(12, 6),
    turnover_anual  numeric(12, 6),
    n_rebalanceos   integer,
    friccion        numeric(8, 6),
    actualizado_en  timestamptz default now()
);

create table if not exists public.backtest_ml_curva (
    fecha          date primary key,
    estrategia     numeric(14, 6),
    estrategia_ml  numeric(14, 6),
    pesos_iguales  numeric(14, 6),
    spy            numeric(14, 6)
);

-- Recomendación por perfil de riesgo (reporte prescriptivo) -----------------------
create table if not exists public.recomendaciones (
    fecha         date not null,
    perfil        text not null,
    fuente        text not null check (fuente in ('llm', 'reglas')),   -- quién redactó el texto
    modelo_llm    text,
    resumen       text not null,
    detalle       jsonb not null,     -- pesos, acciones, explicación por ETF, riesgos y qué vigilar
    entrada_hash  text,               -- huella de los datos enviados al LLM (evita llamadas repetidas)
    creado_en     timestamptz not null default now(),
    primary key (fecha, perfil)
);

-- Bitácora de llamadas al modelo de lenguaje ---------------------------------------
create table if not exists public.llm_log (
    id              bigint generated always as identity primary key,
    creado_en       timestamptz not null default now(),
    dia             date not null default current_date,
    corrida_id      bigint references public.corridas (id) on delete set null,
    perfil          text,
    modelo          text,
    estado          text not null check (estado in ('ok', 'rechazada', 'error', 'limite', 'sin_clave')),
    tokens_entrada  integer,
    tokens_salida   integer,
    latencia_ms     integer,
    mensaje         text
);
create index if not exists idx_llm_log_dia on public.llm_log (dia);

-- Vista para el dashboard: pesos de la última fecha ---------------------------
create or replace view public.v_pesos_actuales
with (security_invoker = true) as
select p.fecha, p.ticker, e.nombre, e.clase_activo,
       p.peso, p.peso_base, p.factor_momentum, p.motivo
from public.pesos_sugeridos p
join public.etfs e on e.ticker = p.ticker
where p.fecha = (select max(fecha) from public.pesos_sugeridos);

-- Comentarios (se ven en el panel de Supabase) --------------------------------
comment on table public.corridas is 'Bitácora de cada ejecución del pipeline (GitHub Actions o manual).';
comment on table public.precios_diarios is 'Precios diarios por ETF, tomados de Yahoo Finance.';
comment on table public.indicadores is 'Retornos, volatilidad, momentum y drawdown por ETF y día.';
comment on table public.pesos_sugeridos is 'Reparto de capital sugerido entre los ETFs en cada fecha.';
comment on table public.backtest_resumen is 'Métricas del backtest: regla ETF2 vs pesos iguales vs solo SPY.';
comment on table public.modelo_predicciones is 'Probabilidad de subida a un mes por ETF y modelo (walk-forward).';
comment on table public.modelo_metricas is 'Métricas fuera de muestra de cada modelo y de las referencias.';
comment on table public.pesos_ml is 'Pesos de la regla con la inclinación del modelo predictivo.';
comment on table public.backtest_ml_resumen is 'Backtest de la regla con y sin modelo, mismo periodo.';
comment on table public.recomendaciones is 'Recomendación por perfil de riesgo: números por reglas, texto por LLM validado.';
comment on table public.llm_log is 'Cada llamada al modelo de lenguaje: estado, tokens y latencia.';

-- Seguridad: lectura pública, escritura solo con la clave secreta ---------------
-- Dos capas: (1) permisos (GRANT): anon y authenticated solo reciben SELECT;
-- (2) RLS activado con una única política de lectura. La clave secreta (rol
-- service_role, guardada solo en GitHub Secrets) se salta RLS y es la única que escribe.
grant usage on schema public to anon, authenticated, service_role;
grant select on all tables in schema public to anon, authenticated;
grant all on all tables in schema public to service_role;

do $$
declare t text;
begin
    foreach t in array array['etfs', 'corridas', 'precios_diarios', 'indicadores',
                             'pesos_sugeridos', 'backtest_resumen', 'backtest_curva',
                             'modelo_predicciones', 'modelo_metricas', 'modelo_importancias',
                             'pesos_ml', 'backtest_ml_resumen', 'backtest_ml_curva',
                             'recomendaciones', 'llm_log']
    loop
        execute format('alter table public.%I enable row level security', t);
        execute format('drop policy if exists lectura_publica on public.%I', t);
        execute format('create policy lectura_publica on public.%I
                        for select to anon, authenticated using (true)', t);
    end loop;
end $$;
