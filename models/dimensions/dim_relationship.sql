with fair_relationships as (
    select
        'fair_' || record_id as relationship_key,
        years_married as years_in_relationship,
        has_children,
        num_children,
        marriage_satisfaction as satisfaction_rating,
        null::double as love_rating,
        null::double as desire_rating,
        null as attachment_style,
        data_source
    from {{ ref('stg_fair__affairs') }}
),

gss_relationships as (
    select
        'gss_' || record_id as relationship_key,
        null::double as years_in_relationship,
        null as has_children,
        null::integer as num_children,
        case
            when marital_happiness_score = 1 then 5.0
            when marital_happiness_score = 2 then 3.0
            when marital_happiness_score = 3 then 1.0
        end as satisfaction_rating,
        null::double as love_rating,
        null::double as desire_rating,
        null as attachment_style,
        data_source
    from {{ ref('stg_gss__infidelity') }}
),

selterman_relationships as (
    select
        'selt_' || record_id as relationship_key,
        relationship_length_months / 12.0 as years_in_relationship,
        null as has_children,
        null::integer as num_children,
        relationship_satisfaction as satisfaction_rating,
        love as love_rating,
        desire as desire_rating,
        attachment_style,
        data_source
    from {{ ref('stg_selterman__predictors') }}
),

reinhardt_relationships as (
    select
        'rein_' || record_id as relationship_key,
        relationship_length_months / 12.0 as years_in_relationship,
        null as has_children,
        null::integer as num_children,
        null::double as satisfaction_rating,
        null::double as love_rating,
        null::double as desire_rating,
        null as attachment_style,
        data_source
    from {{ ref('stg_reinhardt__hexaco') }}
    where in_relationship = true
),

unioned as (
    select * from fair_relationships
    union all
    select * from gss_relationships
    union all
    select * from selterman_relationships
    union all
    select * from reinhardt_relationships
)

select * from unioned
