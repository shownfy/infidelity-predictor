with source as (
    select * from {{ source('raw', 'big_five_raw') }}
),

cleaned as (
    select
        row_number() over () as record_id,
        age,
        gender,
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness,
        'openpsychometrics' as data_source
    from source
    where extraversion is not null
      and agreeableness is not null
      and conscientiousness is not null
      and neuroticism is not null
      and openness is not null
)

select * from cleaned
