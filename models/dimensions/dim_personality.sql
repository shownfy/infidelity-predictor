-- HEXACO 6-factor personality dimension
-- Sources: Reinhardt (full HEXACO), Selterman/GSS/Fair (partial mapping)

with reinhardt_personality as (
    select
        'rein_' || record_id as personality_key,
        honesty_humility,
        emotionality,
        extraversion,
        agreeableness,
        conscientiousness,
        openness,
        data_source
    from {{ ref('stg_reinhardt__hexaco') }}
),

-- For Fair/GSS/Selterman, we don't have direct HEXACO scores
-- We set honesty_humility to null (will be imputed during ML training)
fair_personality as (
    select
        'fair_' || record_id as personality_key,
        null::double as honesty_humility,
        null::double as emotionality,
        null::double as extraversion,
        null::double as agreeableness,
        null::double as conscientiousness,
        null::double as openness,
        data_source
    from {{ ref('stg_fair__affairs') }}
),

gss_personality as (
    select
        'gss_' || record_id as personality_key,
        null::double as honesty_humility,
        null::double as emotionality,
        null::double as extraversion,
        null::double as agreeableness,
        null::double as conscientiousness,
        null::double as openness,
        data_source
    from {{ ref('stg_gss__infidelity') }}
),

selterman_personality as (
    select
        'selt_' || record_id as personality_key,
        null::double as honesty_humility,
        null::double as emotionality,
        null::double as extraversion,
        null::double as agreeableness,
        null::double as conscientiousness,
        null::double as openness,
        data_source
    from {{ ref('stg_selterman__predictors') }}
),

unioned as (
    select * from reinhardt_personality
    union all
    select * from fair_personality
    union all
    select * from gss_personality
    union all
    select * from selterman_personality
)

select * from unioned
