# Validation report

- Rows: 1000 | Columns: 15 | NEET rows: 97 (9.7%)

| Check | Result |
|---|---|
| Row count within 300-1,000 | PASS |
| No missing values | PASS |
| learner_id is unique | PASS |
| Headers are snake_case with no spaces | PASS |
| Categories normalised (lowercase, underscores, no stray spaces) | PASS |
| age between 16 and 24 | PASS |
| attendance_pct between 0 and 100 | PASS |
| household_income >= 0 | PASS |
| duration_neet_months >= 0 | PASS |
| Binary columns contain only 0/1 | PASS |
| NEET label matches rule (not employed AND not in education/training) | PASS |
| NEET learners have duration >= 1, others 0 | PASS |
| neet_type consistent with neet | PASS |
| job_search_frequency is not_applicable only for non-NEET | PASS |
| No commas or quotes in text fields (safe for Db2 delimited import) | PASS |

## Category distributions
- **ethnicity**: white 81.4%, indian 4.2%, black_african_caribbean 3.3%, pakistani 2.9%, mixed_multiple 2.6%, other_asian 1.9%, other_ethnic_group 1.8%, chinese 1.5%, bangladeshi 0.4%
- **qualification_level**: gcse 31.0%, a_level 27.6%, degree 22.5%, none 6.1%, higher_education 5.8%, other 4.1%, dont_know 2.9%
- **neet_type**: not_neet 90.3%, inactive 6.2%, unemployed 3.5%
- **benefit_type**: none 89.0%, uc_searching_for_work 3.1%, uc_no_work_requirements 3.0%, uc_other 2.1%, pip 1.6%, other_benefits 1.2%
- **job_search_frequency**: not_applicable 90.3%, never 3.5%, daily 2.4%, weekly 2.1%, monthly 1.7%
