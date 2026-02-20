with source as (
    select * from {{ source('raw', 'japan_judicial_stats') }}
),

cleaned as (
    select
        year,
        gender,
        age_group,
        motive,
        motive_en,
        filing_rate,
        source as stat_source
    from source
)

select * from cleaned
