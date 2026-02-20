with source as (
    select * from {{ source('raw', 'reinhardt_hexaco') }}
),

cleaned as (
    select
        row_number() over () as record_id,
        age,
        case when gender = 0 then 'male' else 'female' end as gender,
        honesty_humility,
        emotionality,
        extraversion,
        agreeableness,
        conscientiousness,
        openness,
        relationship_dishonesty,
        had_relationship_dishonesty::boolean as had_dishonesty,
        in_relationship::boolean as in_relationship,
        relationship_length_months,
        study_id,
        'reinhardt' as data_source
    from source
)

select * from cleaned
