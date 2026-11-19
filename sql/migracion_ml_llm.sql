-- =============================================================================
-- Migración: tablas del modelo predictivo y del reporte prescriptivo
--
-- Para una base que ya tiene el esquema original (sql/schema.sql). En Supabase:
-- "SQL Editor" -> "New query", pega TODO este archivo y pulsa "Run".
-- Es seguro ejecutarlo más de una vez y no toca los datos existentes.
-- (Una base nueva no lo necesita: schema.sql ya incluye estas tablas.)
-- =============================================================================

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

-- Seguridad: igual que el resto, lectura pública y escritura solo con la clave secreta
grant select on all tables in schema public to anon, authenticated;
grant all on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

do $$
declare t text;
begin
    foreach t in array array['modelo_predicciones', 'modelo_metricas', 'modelo_importancias',
                             'pesos_ml', 'backtest_ml_resumen', 'backtest_ml_curva',
                             'recomendaciones', 'llm_log']
    loop
        execute format('alter table public.%I enable row level security', t);
        execute format('drop policy if exists lectura_publica on public.%I', t);
        execute format('create policy lectura_publica on public.%I
                        for select to anon, authenticated using (true)', t);
    end loop;
end $$;
