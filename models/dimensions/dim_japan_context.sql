with judicial as (
    select
        gender,
        age_group,
        filing_rate as infidelity_filing_rate,
        year,
        'judicial' as stat_type,
        stat_source
    from {{ ref('stg_japan__judicial') }}
    where motive_en = 'infidelity'
      and age_group != 'all'
),

vital as (
    select
        null as gender,
        group_en as age_group,
        divorce_rate,
        year,
        category as stat_type,
        stat_source
    from {{ ref('stg_japan__vital') }}
),

combined as (
    select
        row_number() over () as context_key,
        gender,
        age_group,
        infidelity_filing_rate,
        null::double as divorce_rate,
        year,
        stat_type,
        stat_source
    from judicial

    union all

    select
        row_number() over () + 1000 as context_key,
        gender,
        age_group,
        null::double as infidelity_filing_rate,
        divorce_rate,
        year,
        stat_type,
        stat_source
    from vital
)

select * from combined
