"""
inference.py - score learners with the trained NEET early-warning model.

Command line (run from the repo root; this is what ibm_z/run_inference_on_z.sh calls):
    python model/inference.py --input data/neet_data.csv --output risk_scores.csv

From Python (e.g. the dashboard):
    from model.inference import score
    result = score(df)

Output columns:
    learner_id, risk_score, risk_level,
    top_factor_1, top_factor_2, top_factor_3, suggested_intervention
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "model.pkl"

ID_COL = "learner_id"
NO_ACTION = "No action needed - monitor"

# Suggested intervention for each early-warning factor
INTERVENTIONS = {
    "attendance_pct": "School/college attendance support and engagement check-in",
    "mental_health_condition": "Referral to youth mental health support service",
    "send": "SEND-specific transition support and tailored pathway planning",
    "qualification_level": "Careers guidance and access to re-skilling or qualification programmes",
    "household_income": "Financial support signposting (bursaries, travel and training grants)",
    "age": "Targeted transition support at key age points (post-16 / post-18)",
}


def _factor_label(feature, row):
    """Human-readable description of a risk factor for one learner."""
    if feature == "attendance_pct":
        return f"Low attendance ({row['attendance_pct']:.0f}%)"
    if feature == "mental_health_condition":
        return "Recorded mental health condition"
    if feature == "send":
        return "SEND flag"
    if feature == "qualification_level":
        q = str(row["qualification_level"]).replace("_", " ")
        return f"Qualification level: {q}"
    if feature == "household_income":
        return f"Low household income (GBP {row['household_income']:,.0f})"
    if feature == "age":
        return f"Age {int(row['age'])}"
    return feature


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def _contributions(bundle, X):
    """Per-learner contribution of each ORIGINAL feature to the log-odds of NEET.

    contribution = (transformed value - training mean) * coefficient, summed over
    the one-hot columns that belong to the same original feature.
    Positive = pushes risk up compared with the average learner.
    """
    pipe = bundle["pipeline"]
    X_t = pipe.named_steps["prep"].transform(X)
    coef = pipe.named_steps["clf"].coef_[0]
    contrib = (X_t - bundle["baseline_mean"]) * coef

    names = bundle["feature_names_out"]  # e.g. num__age, cat__qualification_level_gcse
    out = pd.DataFrame(0.0, index=X.index, columns=bundle["features"])
    for j, name in enumerate(names):
        kind, rest = name.split("__", 1)
        if kind == "num":
            orig = rest
        else:
            orig = next(c for c in bundle["categorical"] if rest.startswith(c))
        out[orig] += contrib[:, j]
    return out


def score(df, bundle=None):
    """Return a DataFrame of risk scores and explanations for the learners in df."""
    bundle = bundle or load_bundle()
    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df[feats]
    risk = bundle["pipeline"].predict_proba(X)[:, 1]
    th = bundle["thresholds"]
    level = np.where(risk >= th["high"], "High", np.where(risk >= th["medium"], "Medium", "Low"))

    contrib = _contributions(bundle, X)
    top1, top2, top3, action = [], [], [], []
    for i in range(len(df)):
        row = df.iloc[i]
        c = contrib.iloc[i]
        top = c[c > 0].sort_values(ascending=False).index.tolist()[:3]  # risk-increasing only
        labels = [_factor_label(f, row) for f in top] + [""] * (3 - len(top))
        top1.append(labels[0])
        top2.append(labels[1])
        top3.append(labels[2])
        action.append(INTERVENTIONS[top[0]] if top else NO_ACTION)

    result = pd.DataFrame(
        {
            ID_COL: df[ID_COL].values if ID_COL in df.columns else np.arange(len(df)),
            "risk_score": np.round(risk, 4),
            "risk_level": level,
            "top_factor_1": top1,
            "top_factor_2": top2,
            "top_factor_3": top3,
            "suggested_intervention": action,
        }
    )
    # Low-risk learners get no intervention
    result.loc[result["risk_level"] == "Low", "suggested_intervention"] = NO_ACTION
    return result


def main():
    ap = argparse.ArgumentParser(description="Score learners for NEET risk")
    ap.add_argument("--input", default=str(ROOT / "data" / "neet_data.csv"))
    ap.add_argument("--output", default=str(ROOT / "risk_scores.csv"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    result = score(df, load_bundle(args.model))
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} learners -> {args.output}")
    print(result["risk_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
