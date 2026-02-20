with source as (
    select * from {{ source('raw', 'japan_vital_stats') }}
),

cleaned as (
    select
        year,
        category,
        "group" as group_name,
        group_en,
        divorce_rate,
        source as stat_source
    from source
)

select * from cleaned
