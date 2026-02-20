with source as (
    select * from {{ source('raw', 'selterman_predictors') }}
),

cleaned as (
    select
        row_number() over () as record_id,
        age,
        case when gender = 0 then 'male' else 'female' end as gender,
        relationship_satisfaction,
        love,
        desire,
        relationship_length_months,
        attachment_anxiety,
        attachment_avoidance,
        case
            when attachment_avoidance <= 2.5 and attachment_anxiety <= 2.5 then 'secure'
            when attachment_anxiety > 2.5 and attachment_avoidance <= 2.5 then 'anxious'
            when attachment_avoidance > 2.5 and attachment_anxiety <= 2.5 then 'avoidant'
            else 'fearful_avoidant'
        end as attachment_style,
        had_infidelity::boolean as had_affair,
        infidelity_type,
        'selterman' as data_source
    from source
)

select * from cleaned
