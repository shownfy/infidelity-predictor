-- Personality trait distribution statistics for normalization
-- Big Five from Open Psychometrics, H-H from Reinhardt

with big_five_stats as (
    select
        'extraversion' as trait_name,
        'big_five' as model_type,
        avg(extraversion) as mean_score,
        stddev(extraversion) as std_score,
        percentile_cont(0.25) within group (order by extraversion) as p25,
        percentile_cont(0.50) within group (order by extraversion) as p50,
        percentile_cont(0.75) within group (order by extraversion) as p75,
        count(*) as sample_size
    from {{ ref('stg_openpsych__big_five') }}

    union all

    select
        'agreeableness', 'big_five',
        avg(agreeableness), stddev(agreeableness),
        percentile_cont(0.25) within group (order by agreeableness),
        percentile_cont(0.50) within group (order by agreeableness),
        percentile_cont(0.75) within group (order by agreeableness),
        count(*)
    from {{ ref('stg_openpsych__big_five') }}

    union all

    select
        'conscientiousness', 'big_five',
        avg(conscientiousness), stddev(conscientiousness),
        percentile_cont(0.25) within group (order by conscientiousness),
        percentile_cont(0.50) within group (order by conscientiousness),
        percentile_cont(0.75) within group (order by conscientiousness),
        count(*)
    from {{ ref('stg_openpsych__big_five') }}

    union all

    select
        'neuroticism', 'big_five',
        avg(neuroticism), stddev(neuroticism),
        percentile_cont(0.25) within group (order by neuroticism),
        percentile_cont(0.50) within group (order by neuroticism),
        percentile_cont(0.75) within group (order by neuroticism),
        count(*)
    from {{ ref('stg_openpsych__big_five') }}

    union all

    select
        'openness', 'big_five',
        avg(openness), stddev(openness),
        percentile_cont(0.25) within group (order by openness),
        percentile_cont(0.50) within group (order by openness),
        percentile_cont(0.75) within group (order by openness),
        count(*)
    from {{ ref('stg_openpsych__big_five') }}
),

hexaco_hh_stats as (
    select
        'honesty_humility' as trait_name,
        'hexaco' as model_type,
        avg(honesty_humility) as mean_score,
        stddev(honesty_humility) as std_score,
        percentile_cont(0.25) within group (order by honesty_humility) as p25,
        percentile_cont(0.50) within group (order by honesty_humility) as p50,
        percentile_cont(0.75) within group (order by honesty_humility) as p75,
        count(*) as sample_size
    from {{ ref('stg_reinhardt__hexaco') }}
),

combined as (
    select * from big_five_stats
    union all
    select * from hexaco_hh_stats
)

select
    row_number() over () as reference_key,
    *
from combined
