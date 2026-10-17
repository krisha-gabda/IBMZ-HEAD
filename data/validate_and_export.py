"""
12:00 slot - clean/validate the dataset, export a Db2-ready CSV, and write the
data guide.  Run AFTER generate_neet_dataset.py.

Outputs: validation_report.md, neet_dataset_db2.csv, DATA_GUIDE.md
"""
import pandas as pd

df = pd.read_csv("neet_dataset.csv")
dd = pd.read_csv("data_dictionary.csv")
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
binary = ["send", "mental_health_condition", "in_employment", "in_education_training", "neet"]

# ------------------------------------------------------------------ 12:00-12:30 VALIDATE
checks = []
def check(name, ok):
    checks.append((name, bool(ok)))

normalised = all(
    df[c].eq(df[c].str.strip().str.lower().str.replace(r"\s+", "_", regex=True)).all()
    for c in cat_cols)
label = ((df.in_employment == 0) & (df.in_education_training == 0)).astype(int)

check("Row count within 300-1,000", 300 <= len(df) <= 1000)
check("No missing values", df.isna().sum().sum() == 0)
check("learner_id is unique", df.learner_id.is_unique)
check("Headers are snake_case with no spaces", all(c == c.lower() and " " not in c for c in df.columns))
check("Categories normalised (lowercase, underscores, no stray spaces)", normalised)
check("age between 16 and 24", df.age.between(16, 24).all())
check("attendance_pct between 0 and 100", df.attendance_pct.between(0, 100).all())
check("household_income >= 0", (df.household_income >= 0).all())
check("duration_neet_months >= 0", (df.duration_neet_months >= 0).all())
check("Binary columns contain only 0/1", all(df[c].isin([0, 1]).all() for c in binary))
check("NEET label matches rule (not employed AND not in education/training)", label.eq(df.neet).all())
check("NEET learners have duration >= 1, others 0",
      (((df.neet == 1) & (df.duration_neet_months >= 1)) | ((df.neet == 0) & (df.duration_neet_months == 0))).all())
check("neet_type consistent with neet", ((df.neet == 1) == (df.neet_type != "not_neet")).all())
check("job_search_frequency is not_applicable only for non-NEET",
      ((df.job_search_frequency == "not_applicable") == (df.neet == 0)).all())

# ------------------------------------------------------------------ DB2 EXPORT
db2 = df.copy()
db2["household_income"] = db2["household_income"].round().astype(int)
unsafe = db2[cat_cols].apply(lambda s: s.str.contains(r"[,\"']")).any().any()
check("No commas or quotes in text fields (safe for Db2 delimited import)", not unsafe)
db2.to_csv("neet_dataset_db2.csv", index=False, lineterminator="\n", na_rep="")

# ------------------------------------------------------------------ VALIDATION REPORT
n_neet = int(df.neet.sum())
lines = ["# Validation report", "",
         f"- Rows: {len(df)} | Columns: {df.shape[1]} | NEET rows: {n_neet} ({df.neet.mean():.1%})", "",
         "| Check | Result |", "|---|---|"]
lines += [f"| {n} | {'PASS' if ok else 'FAIL'} |" for n, ok in checks]
lines += ["", "## Category distributions"]
for c in cat_cols:
    dist = df[c].value_counts(normalize=True).round(3)
    lines.append(f"- **{c}**: " + ", ".join(f"{k} {v:.1%}" for k, v in dist.items()))
open("validation_report.md", "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ 12:30-13:00 DATA GUIDE
h = ["# Data guide: NEET label, columns and usage", "",
     "## NEET label logic", "",
     "A learner is NEET (`neet = 1`) if they are **not in employment AND not in education or training**, aged 16-24.", "",
     "```python",
     "neet = int(in_employment == 0 and in_education_training == 0)",
     "```", "",
     "## Column roles (for anyone modelling or analysing this data)", "",
     "- **Target:** `neet`",
     "- **Exclude (identifier):** `learner_id`",
     "- **Exclude (leakage):** `in_employment`, `in_education_training` (they define the label), "
     "`neet_type` (derived from the label), `duration_neet_months` (always 0 for non-NEETs), "
     "`job_search_frequency` (always `not_applicable` for non-NEETs)",
     "- **Safe features:** `age`, `ethnicity`, `send`, `mental_health_condition`, `qualification_level`, "
     "`attendance_pct`, `benefit_type`, `household_income`", "",
     "**Alternative target (NEET rows only):** `long_term_neet = duration_neet_months >= 12`. "
     "If you use it, exclude `duration_neet_months` (it defines the target); `job_search_frequency` becomes a valid feature. "
     "There are few NEET rows, so treat results as indicative.", "",
     "## Practical notes", "",
     f"- Only {n_neet} of {len(df)} rows are NEET, so classes are imbalanced. Use a stratified split, "
     "class weights, and report recall and precision-recall rather than accuracy alone.",
     "- `duration_neet_months` is a proxy: the annex says its source measure (time out of full-time education or employment) does not measure time NEET.",
     "- The overall NEET rate follows pooled 2018-2025 UK figures (about 12%); the latest quarterly rate is higher (13.0%, April-June 2026).",
     "- The data is synthetic. The model recovers relationships built into the generator and annex proportions; "
     "it does not discover real-world causes. State this in the README and video.", "",
     "## Data dictionary", "",
     "| Column | Type | Allowed values | Description | Source |", "|---|---|---|---|---|"]
h += [f"| {r.column} | {r.type} | {r.allowed_values} | {r.description} | {r.source} |" for r in dd.itertuples()]
open("DATA_GUIDE.md", "w").write("\n".join(h) + "\n")

print(open("validation_report.md").read()[:1800])
assert all(ok for _, ok in checks), "Some checks failed"
