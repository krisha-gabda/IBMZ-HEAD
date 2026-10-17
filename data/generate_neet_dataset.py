"""
Synthetic NEET dataset generator - IBM Z Datathon (Data Lead).

Proportions come from the DWP "Young People and Work" interim report analytical
annex data tables (table numbers are cited in data_dictionary.csv).
Anything the annex does NOT cover is marked ASSUMPTION in this file and in the
data dictionary. All data is synthetic: no real individuals are represented.

Run:  python generate_neet_dataset.py   ->  neet_dataset.csv, data_dictionary.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000  # brief asks for 300-1,000 rows


def pick(options, probs):
    p = np.array(probs, dtype=float)
    return rng.choice(options, p=p / p.sum())


# ------------------------------------------------------------------ SOURCE FIGURES
# Table 1: status by ethnicity -> [education/training, employment, unemployed, inactive]
# (unemployed + inactive = NEET)
T1 = {
    "white":                   [0.563, 0.315, 0.043, 0.075],
    "mixed_multiple":          [0.648, 0.223, 0.054, 0.067],
    "indian":                  [0.689, 0.220, 0.037, 0.048],
    "pakistani":               [0.685, 0.179, 0.062, 0.071],
    "bangladeshi":             [0.675, 0.221, 0.041, 0.062],
    "chinese":                 [0.779, 0.142, 0.039, 0.035],
    "other_asian":             [0.699, 0.186, 0.050, 0.063],
    "black_african_caribbean": [0.695, 0.168, 0.065, 0.066],
    "other_ethnic_group":      [0.695, 0.169, 0.048, 0.083],
}
# ASSUMPTION: share of 16-24s by ethnicity (not in annex)
ETH_SHARE = {"white": 0.80, "mixed_multiple": 0.04, "indian": 0.04, "pakistani": 0.03,
             "bangladeshi": 0.01, "chinese": 0.01, "other_asian": 0.02,
             "black_african_caribbean": 0.03, "other_ethnic_group": 0.02}

# Table 24: highest qualification of NEETs
T24 = {"degree": 0.155, "higher_education": 0.023, "a_level": 0.214, "gcse": 0.293,
       "other": 0.107, "none": 0.181, "dont_know": 0.027}
# ASSUMPTION: qualification of non-NEETs (annex only covers NEETs)
Q_NON_NEET = {"degree": 0.25, "higher_education": 0.05, "a_level": 0.30, "gcse": 0.28,
              "other": 0.05, "none": 0.05, "dont_know": 0.02}

# Table 21: NEET duration [under 1 year, 1 year or more] by qualification ("unknown" dropped)
T21 = {"degree": (0.575, 0.175), "higher_education": (0.575, 0.175),
       "a_level": (0.415, 0.441), "gcse": (0.336, 0.521), "other": (0.177, 0.711),
       "none": (0.158, 0.680), "dont_know": (0.348, 0.486)}   # dont_know uses the Total row

# Table 13: benefit status of NEETs (England, 18-24)
T13 = {"none": 0.46, "uc_no_work_requirements": 0.20, "uc_searching_for_work": 0.19,
       "uc_other": 0.07, "pip": 0.04, "other_benefits": 0.03}
# ASSUMPTION: benefit status of non-NEETs
B_NON_NEET = {"none": 0.93, "uc_no_work_requirements": 0.01, "uc_searching_for_work": 0.02,
              "uc_other": 0.02, "pip": 0.01, "other_benefits": 0.01}

# Table 25: 64,315 of 879,814 NEETs aged 16-24 are aged 16-17
P_NEET_16_17 = 64315 / 879814

# Table 57b: 81% of 18-24 UC LCWRA claimants have a recorded mental health or
# neurodevelopmental condition. Used as a PROXY for NEETs on UC (no work reqs) / PIP.
P_MH_CLAIMANT = 0.81
P_MH_NEET_OTHER = 0.25   # ASSUMPTION
P_MH_NON_NEET = 0.15     # ASSUMPTION
P_SEND_NEET = 0.28       # ASSUMPTION
P_SEND_NON_NEET = 0.16   # ASSUMPTION

Q_INCOME_ADJ = {"degree": 0.15, "higher_education": 0.05, "a_level": 0.0, "gcse": -0.10,
                "other": -0.10, "none": -0.25, "dont_know": 0.0}   # ASSUMPTION

# ------------------------------------------------------------------ GENERATE
rows = []
for i in range(1, N + 1):
    eth = pick(list(ETH_SHARE), list(ETH_SHARE.values()))
    status = pick(["education_training", "employment", "unemployed", "inactive"], T1[eth])
    neet = int(status in ("unemployed", "inactive"))

    if neet:
        age = int(rng.integers(16, 18)) if rng.random() < P_NEET_16_17 else int(rng.integers(18, 25))
        qual = pick(list(T24), list(T24.values()))
        benefit = pick(list(T13), list(T13.values()))
        under, over = T21[qual]
        if rng.random() < under / (under + over):
            months = int(rng.integers(1, 12))                       # ASSUMPTION: spread within bucket
        else:
            months = int(min(60, 12 + rng.exponential(12)))         # ASSUMPTION: spread within bucket
        if benefit == "uc_searching_for_work":                      # ASSUMPTION (informed by Tables 13, 2)
            js = pick(["weekly", "daily"], [0.35, 0.65])
        elif status == "unemployed":
            js = pick(["never", "monthly", "weekly", "daily"], [0.10, 0.25, 0.40, 0.25])
        else:
            js = pick(["never", "monthly", "weekly", "daily"], [0.60, 0.25, 0.10, 0.05])
        on_health_benefit = benefit in ("uc_no_work_requirements", "pip")
        mh = int(rng.random() < (P_MH_CLAIMANT if on_health_benefit else P_MH_NEET_OTHER))
        send = int(rng.random() < P_SEND_NEET)
        attendance = float(np.clip(rng.beta(5, 2) * 100, 0, 100).round(1))
    else:
        age = int(rng.integers(16, 25))
        qual = pick(list(Q_NON_NEET), list(Q_NON_NEET.values()))
        benefit = pick(list(B_NON_NEET), list(B_NON_NEET.values()))
        months, js = 0, "not_applicable"
        mh = int(rng.random() < P_MH_NON_NEET)
        send = int(rng.random() < P_SEND_NON_NEET)
        attendance = float(np.clip(rng.beta(8, 1.5) * 100, 0, 100).round(1))

    # ASSUMPTION: household income (GBP/yr), lower for NEET and low qualification
    income = int(round(float(rng.lognormal(np.log(28000) - 0.25 * neet + Q_INCOME_ADJ[qual], 0.45)), -2))

    rows.append({
        "learner_id": i, "age": age, "ethnicity": eth, "send": send,
        "mental_health_condition": mh, "qualification_level": qual,
        "attendance_pct": attendance,
        "in_employment": int(status == "employment"),
        "in_education_training": int(status == "education_training"),
        "neet_type": status if neet else "not_neet",
        # LABEL LOGIC: neet = 1 if NOT in employment AND NOT in education/training
        "neet": neet, "duration_neet_months": months, "benefit_type": benefit,
        "job_search_frequency": js, "household_income": income,
    })

df = pd.DataFrame(rows)


# ------------------------------------------------------------------ CLEAN + VALIDATE
def clean(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        frame[col] = (frame[col].str.strip().str.lower()
                      .str.replace(r"[\s/-]+", "_", regex=True))
    frame = frame.dropna()
    assert frame["age"].between(16, 24).all(), "age out of range"
    assert (frame["household_income"] >= 0).all(), "negative income"
    assert (frame["duration_neet_months"] >= 0).all(), "negative duration"
    assert frame["attendance_pct"].between(0, 100).all(), "attendance out of range"
    assert frame["learner_id"].is_unique, "duplicate ids"
    check = (frame["in_employment"] == 0) & (frame["in_education_training"] == 0)
    assert (check.astype(int) == frame["neet"]).all(), "NEET label inconsistent"
    return frame


df = clean(df)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
df.to_csv("neet_dataset.csv", index=False)

# ------------------------------------------------------------------ DATA DICTIONARY
meta = {
    "learner_id": ("Unique synthetic learner identifier", "Generated"),
    "age": ("Age in years (16-24)", "Table 25 (7% of NEETs aged 16-17); non-NEET uniform = ASSUMPTION"),
    "ethnicity": ("Ethnic group (annex categories)", "Categories from Table 1; population shares = ASSUMPTION"),
    "send": ("Special educational needs/disability flag (1 = yes)", "ASSUMPTION - not in annex"),
    "mental_health_condition": ("Recorded mental health or neurodevelopmental condition (1 = yes)",
                                "Table 57b (81% of 18-24 UC LCWRA claimants with a health condition cited at WCA, 2025) used as proxy for NEETs on UC (no work reqs)/PIP; rest ASSUMPTION"),
    "qualification_level": ("Highest qualification held", "Table 24 (composition of NEETs by highest qualification); non-NEET = ASSUMPTION"),
    "attendance_pct": ("Attendance in last education year (%)", "ASSUMPTION - not in annex"),
    "in_employment": ("Currently in employment (1 = yes)", "Table 1 (status of all 16-24s by ethnicity, UK, pooled 2018-2025)"),
    "in_education_training": ("Currently in education or training (1 = yes)", "Table 1 (status of all 16-24s by ethnicity, UK, pooled 2018-2025)"),
    "neet_type": ("not_neet, unemployed or inactive", "Table 1 (unemployed vs inactive)"),
    "neet": ("Label: 1 if not in employment AND not in education/training", "Derived; ONS definition in Annex section 29 (not in education/training and not in employment)"),
    "duration_neet_months": ("Approximate months out of full-time education or employment, used as a proxy for time NEET (0 if not NEET)",
                             "Table 21 (under 1 year vs 1+ years by qualification). Annex section 8 says this is an approximation, not time NEET; "
                             "unknowns dropped; spread within bucket = ASSUMPTION"),
    "benefit_type": ("Benefit status", "Table 13 (NEETs); non-NEET = ASSUMPTION"),
    "job_search_frequency": ("How often the learner searches for work (NEET only)",
                             "ASSUMPTION - informed by Tables 13 and 2"),
    "household_income": ("Annual household income (GBP)", "ASSUMPTION - not in annex"),
}
dd = []
for col in df.columns:
    numeric = pd.api.types.is_numeric_dtype(df[col])
    allowed = (f"{df[col].min()} to {df[col].max()}" if numeric and df[col].nunique() > 10
               else " | ".join(sorted(map(str, df[col].unique()))))
    dd.append({"column": col, "type": ("integer" if pd.api.types.is_integer_dtype(df[col]) else "decimal" if pd.api.types.is_float_dtype(df[col]) else "text"), "allowed_values": allowed,
               "description": meta[col][0], "source": meta[col][1]})
pd.DataFrame(dd).to_csv("data_dictionary.csv", index=False)

print(f"Rows: {len(df)} | NEET rate: {df['neet'].mean():.1%} | NEET rows: {df['neet'].sum()}")
