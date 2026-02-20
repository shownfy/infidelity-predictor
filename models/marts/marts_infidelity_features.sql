-- ML training wide table: dim + fct joined
-- This table is the direct input for train_model.py

with facts as (
    select * from {{ ref('fct_infidelity') }}
),

persons as (
    select * from {{ ref('dim_person') }}
),

personalities as (
    select * from {{ ref('dim_personality') }}
),

relationships as (
    select * from {{ ref('dim_relationship') }}
),

joined as (
    select
        -- Keys
        facts.infidelity_key,
        facts.data_source,

        -- Target
        facts.had_affair,
        facts.affair_intensity,
        facts.dishonesty_score,

        -- Person attributes
        persons.gender,
        persons.age,
        persons.age_group,
        persons.education_level,
        persons.education_years,
        persons.religiousness,
        persons.occupation,

        -- Personality (HEXACO)
        personalities.honesty_humility,
        personalities.emotionality,
        personalities.extraversion,
        personalities.agreeableness,
        personalities.conscientiousness,
        personalities.openness,

        -- Relationship
        relationships.years_in_relationship,
        relationships.has_children,
        relationships.num_children,
        relationships.satisfaction_rating,
        relationships.love_rating,
        relationships.desire_rating,
        relationships.attachment_style

    from facts
    left join persons on facts.person_key = persons.person_key
    left join personalities on facts.personality_key = personalities.personality_key
    left join relationships on facts.relationship_key = relationships.relationship_key
)

select * from joined
