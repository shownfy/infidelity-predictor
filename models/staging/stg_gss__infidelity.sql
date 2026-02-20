with source as (
    select * from {{ source('raw', 'gss_infidelity') }}
),

cleaned as (
    select
        row_number() over () as record_id,
        year as survey_year,
        age,
        case when sex = 1 then 'male' else 'female' end as gender,
        educ as education_years,
        case
            when educ <= 12 then 'high_school'
            when educ <= 16 then 'college'
            else 'graduate'
        end as education_level,
        relig as religion_code,
        attend as religious_attendance,
        case
            when marital = 1 then 'married'
            when marital = 2 then 'widowed'
            when marital = 3 then 'divorced'
            when marital = 4 then 'separated'
            when marital = 5 then 'never_married'
        end as marital_status,
        case
            when hapmar = 1 then 'very_happy'
            when hapmar = 2 then 'pretty_happy'
            when hapmar = 3 then 'not_too_happy'
        end as marital_happiness,
        hapmar as marital_happiness_score,
        case when evstray = 1 then true when evstray = 2 then false end as had_affair,
        xmarsex as extramarital_attitude,
        partners as num_partners_last_year,
        'gss' as data_source
    from source
    where marital in (1, 2, 3, 4)  -- ever married only
      and evstray in (1, 2)         -- valid responses only
)

select * from cleaned
