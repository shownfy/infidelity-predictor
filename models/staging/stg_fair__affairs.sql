with source as (
    select * from {{ source('raw', 'fair_affairs_raw') }}
),

cleaned as (
    select
        row_number() over () as record_id,
        age,
        case
            when rate_marriage = 1 then 'very_unhappy'
            when rate_marriage = 2 then 'somewhat_unhappy'
            when rate_marriage = 3 then 'average'
            when rate_marriage = 4 then 'happier_than_average'
            when rate_marriage = 5 then 'very_happy'
        end as marriage_satisfaction_label,
        rate_marriage as marriage_satisfaction,
        yrs_married as years_married,
        children as num_children,
        children > 0 as has_children,
        religious as religiousness,
        educ as education_years,
        case
            when educ <= 12 then 'high_school'
            when educ <= 16 then 'college'
            else 'graduate'
        end as education_level,
        occupation,
        occupation_husb as occupation_spouse,
        affairs as affair_time_spent,
        affairs > 0 as had_affair,
        'fair' as data_source
    from source
)

select * from cleaned
