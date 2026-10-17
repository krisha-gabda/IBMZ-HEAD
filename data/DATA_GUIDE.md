# Data guide: NEET label, columns and usage

## NEET label logic

A learner is NEET (`neet = 1`) if they are **not in employment AND not in education or training**, aged 16-24.

```python
neet = int(in_employment == 0 and in_education_training == 0)
```

## Column roles (for anyone modelling or analysing this data)

- **Target:** `neet`
- **Exclude (identifier):** `learner_id`
- **Exclude (leakage):** `in_employment`, `in_education_training` (they define the label), `neet_type` (derived from the label), `duration_neet_months` (always 0 for non-NEETs), `job_search_frequency` (always `not_applicable` for non-NEETs)
- **Safe features:** `age`, `ethnicity`, `send`, `mental_health_condition`, `qualification_level`, `attendance_pct`, `benefit_type`, `household_income`

**Alternative target (NEET rows only):** `long_term_neet = duration_neet_months >= 12`. If you use it, exclude `duration_neet_months` (it defines the target); `job_search_frequency` becomes a valid feature. There are few NEET rows, so treat results as indicative.

## Practical notes

- Only 97 of 1000 rows are NEET, so classes are imbalanced. Use a stratified split, class weights, and report recall and precision-recall rather than accuracy alone.
- `duration_neet_months` is a proxy: the annex says its source measure (time out of full-time education or employment) does not measure time NEET.
- The overall NEET rate follows pooled 2018-2025 UK figures (about 12%); the latest quarterly rate is higher (13.0%, April-June 2026).
- The data is synthetic. The model recovers relationships built into the generator and annex proportions; it does not discover real-world causes. State this in the README and video.

## Data dictionary

| Column | Type | Allowed values | Description | Source |
|---|---|---|---|---|
| learner_id | integer | 1 to 1000 | Unique synthetic learner identifier | Generated |
| age | integer | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | Age in years (16-24) | Table 25 (7% of NEETs aged 16-17); non-NEET uniform = ASSUMPTION |
| ethnicity | text | bangladeshi | black_african_caribbean | chinese | indian | mixed_multiple | other_asian | other_ethnic_group | pakistani | white | Ethnic group (annex categories) | Categories from Table 1; population shares = ASSUMPTION |
| send | integer | 0 | 1 | Special educational needs/disability flag (1 = yes) | ASSUMPTION - not in annex |
| mental_health_condition | integer | 0 | 1 | Recorded mental health or neurodevelopmental condition (1 = yes) | Table 57b (81% of 18-24 UC LCWRA claimants with a health condition cited at WCA, 2025) used as proxy for NEETs on UC (no work reqs)/PIP; rest ASSUMPTION |
| qualification_level | text | a_level | degree | dont_know | gcse | higher_education | none | other | Highest qualification held | Table 24 (composition of NEETs by highest qualification); non-NEET = ASSUMPTION |
| attendance_pct | decimal | 31.1 to 99.9 | Attendance in last education year (%) | ASSUMPTION - not in annex |
| in_employment | integer | 0 | 1 | Currently in employment (1 = yes) | Table 1 (status of all 16-24s by ethnicity, UK, pooled 2018-2025) |
| in_education_training | integer | 0 | 1 | Currently in education or training (1 = yes) | Table 1 (status of all 16-24s by ethnicity, UK, pooled 2018-2025) |
| neet_type | text | inactive | not_neet | unemployed | not_neet, unemployed or inactive | Table 1 (unemployed vs inactive) |
| neet | integer | 0 | 1 | Label: 1 if not in employment AND not in education/training | Derived; ONS definition in Annex section 29 (not in education/training and not in employment) |
| duration_neet_months | integer | 0 to 60 | Approximate months out of full-time education or employment, used as a proxy for time NEET (0 if not NEET) | Table 21 (under 1 year vs 1+ years by qualification). Annex section 8 says this is an approximation, not time NEET; unknowns dropped; spread within bucket = ASSUMPTION |
| benefit_type | text | none | other_benefits | pip | uc_no_work_requirements | uc_other | uc_searching_for_work | Benefit status | Table 13 (NEETs); non-NEET = ASSUMPTION |
| job_search_frequency | text | daily | monthly | never | not_applicable | weekly | How often the learner searches for work (NEET only) | ASSUMPTION - informed by Tables 13 and 2 |
| household_income | integer | 3000 to 113400 | Annual household income (GBP) | ASSUMPTION - not in annex |
