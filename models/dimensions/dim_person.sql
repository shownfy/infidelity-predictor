with fair_persons as (
    select
        'fair_' || record_id as person_key,
        null as gender,
        age,
        education_level,
        education_years,
        religiousness,
        occupation,
        data_source
    from {{ ref('stg_fair__affairs') }}
),

gss_persons as (
    select
        'gss_' || record_id as person_key,
        gender,
        age,
        education_level,
        education_years,
        religious_attendance as religiousness,
        null as occupation,
        data_source
    from {{ ref('stg_gss__infidelity') }}
),

selterman_persons as (
    select
        'selt_' || record_id as person_key,
        gender,
        age,
        null as education_level,
        null as education_years,
        null as religiousness,
        null as occupation,
        data_source
    from {{ ref('stg_selterman__predictors') }}
),

reinhardt_persons as (
    select
        'rein_' || record_id as person_key,
        gender,
        age,
        null as education_level,
        null as education_years,
        null as religiousness,
        null as occupation,
        data_source
    from {{ ref('stg_reinhardt__hexaco') }}
),

unioned as (
    select * from fair_persons
    union all
    select * from gss_persons
    union all
    select * from selterman_persons
    union all
    select * from reinhardt_persons
)

select
    person_key,
    gender,
    age,
    case
        when age < 25 then '18-24'
        when age < 30 then '25-29'
        when age < 35 then '30-34'
        when age < 40 then '35-39'
        when age < 45 then '40-44'
        when age < 50 then '45-49'
        when age < 55 then '50-54'
        when age < 60 then '55-59'
        else '60+'
    end as age_group,
    education_level,
    education_years,
    religiousness,
    occupation,
    data_source
from unioned
