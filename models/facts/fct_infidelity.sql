with fair_facts as (
    select
        'fair_' || record_id as infidelity_key,
        'fair_' || record_id as person_key,
        'fair_' || record_id as personality_key,
        'fair_' || record_id as relationship_key,
        had_affair,
        affair_time_spent as affair_intensity,
        null::double as dishonesty_score,
        null as infidelity_type,
        data_source
    from {{ ref('stg_fair__affairs') }}
),

gss_facts as (
    select
        'gss_' || record_id as infidelity_key,
        'gss_' || record_id as person_key,
        'gss_' || record_id as personality_key,
        'gss_' || record_id as relationship_key,
        had_affair,
        null::double as affair_intensity,
        null::double as dishonesty_score,
        null as infidelity_type,
        data_source
    from {{ ref('stg_gss__infidelity') }}
),

selterman_facts as (
    select
        'selt_' || record_id as infidelity_key,
        'selt_' || record_id as person_key,
        'selt_' || record_id as personality_key,
        'selt_' || record_id as relationship_key,
        had_affair,
        null::double as affair_intensity,
        null::double as dishonesty_score,
        infidelity_type,
        data_source
    from {{ ref('stg_selterman__predictors') }}
),

reinhardt_facts as (
    select
        'rein_' || record_id as infidelity_key,
        'rein_' || record_id as person_key,
        'rein_' || record_id as personality_key,
        'rein_' || record_id as relationship_key,
        had_dishonesty as had_affair,
        null::double as affair_intensity,
        relationship_dishonesty as dishonesty_score,
        null as infidelity_type,
        data_source
    from {{ ref('stg_reinhardt__hexaco') }}
    where in_relationship = true
),

unioned as (
    select * from fair_facts
    union all
    select * from gss_facts
    union all
    select * from selterman_facts
    union all
    select * from reinhardt_facts
)

select * from unioned
