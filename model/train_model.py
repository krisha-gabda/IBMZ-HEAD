"""
train_model.py - train the NEET early-warning model.

Run from the repo root:
    python model/train_model.py                 # read learners from Db2 (default)
    python model/train_model.py --source csv    # read data/neet_data.csv instead

Outputs:
    model/model.pkl      model bundle (pipeline + thresholds + baselines)
    model/metrics.json   evaluation results (AUC, recall, fairness audit)

Design notes
------------
* Only EARLY-WARNING features are used. Columns that are outcomes of being NEET
  (in_employment, in_education_training, neet_type, duration_neet_months,
  benefit_type, job_search_frequency) would leak the label (AUC = 1.0) and are
  deliberately excluded.
* ethnicity is NOT a model feature. It is used only for a fairness audit.
* Logistic regression is the deployed model because its coefficients are directly
  explainable; a gradient-boosting model is trained only as a comparison.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # works both as `python model/train_model.py` and as a package import
    from model import db_utils
except ImportError:
    import db_utils

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "neet_data.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"

TARGET = "neet"
NUMERIC = ["age", "attendance_pct", "household_income", "send", "mental_health_condition"]
CATEGORICAL = ["qualification_level"]
FEATURES = NUMERIC + CATEGORICAL
SENSITIVE = "ethnicity"  # audit only, never a feature

# Risk-level cut-offs as percentiles of the TRAINING score distribution
MEDIUM_PCT, HIGH_PCT = 70, 90
SEED = 42


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_logreg():
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )


def build_gbm():
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)),
        ]
    )


def recall_at_top(y_true, scores, top_frac):
    """Share of true NEETs captured if we flag the top `top_frac` highest-risk people."""
    k = max(1, int(round(len(scores) * top_frac)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    return float(recall_score(y_true, flagged))


def main():
    ap = argparse.ArgumentParser(description="Train the NEET early-warning model")
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    args = ap.parse_args()

    df = db_utils.read_learners() if args.source == "db" else pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    print(f"Loaded {len(df)} rows | NEET rate = {y.mean():.1%}")

    # ---- 1. Cross-validated comparison -------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_logreg = cross_val_score(build_logreg(), X, y, cv=cv, scoring="roc_auc")
    cv_gbm = cross_val_score(build_gbm(), X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold AUC  logistic regression: {cv_logreg.mean():.3f} (+/- {cv_logreg.std():.3f})")
    print(f"5-fold AUC  gradient boosting  : {cv_gbm.mean():.3f} (+/- {cv_gbm.std():.3f})")

    # ---- 2. Hold-out evaluation --------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = build_logreg().fit(X_tr, y_tr)
    te_scores = model.predict_proba(X_te)[:, 1]
    holdout = {
        "auc": float(roc_auc_score(y_te, te_scores)),
        "recall_top_10pct": recall_at_top(y_te.values, te_scores, 0.10),
        "recall_top_30pct": recall_at_top(y_te.values, te_scores, 0.30),
    }
    print(
        f"Hold-out AUC = {holdout['auc']:.3f} | recall if we flag top 10% = "
        f"{holdout['recall_top_10pct']:.0%}, top 30% = {holdout['recall_top_30pct']:.0%}"
    )

    # ---- 3. Fairness audit (ethnicity not used by the model) ----------------
    # Out-of-fold scores on ALL rows so small groups get more samples.
    oof = cross_val_predict(build_logreg(), X, y, cv=cv, method="predict_proba")[:, 1]
    cut = np.percentile(oof, HIGH_PCT)
    audit = pd.DataFrame(
        {"group": df[SENSITIVE], "neet": y, "score": oof, "flagged_high": oof >= cut}
    )
    fairness = (
        audit.groupby("group")
        .agg(
            n=("neet", "size"),
            actual_neet_rate=("neet", "mean"),
            mean_risk_score=("score", "mean"),
            high_risk_flag_rate=("flagged_high", "mean"),
        )
        .round(3)
    )
    print("\nFairness audit (out-of-fold; groups with n<30 are indicative only):")
    print(fairness.to_string())

    # ---- 4. Final model on ALL data + bundle -------------------------------
    final = build_logreg().fit(X, y)
    train_scores = final.predict_proba(X)[:, 1]
    prep = final.named_steps["prep"]
    X_t = prep.transform(X)
    bundle = {
        "pipeline": final,
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "feature_names_out": list(prep.get_feature_names_out()),
        "baseline_mean": X_t.mean(axis=0),  # reference point for explanations
        "thresholds": {
            "medium": float(np.percentile(train_scores, MEDIUM_PCT)),
            "high": float(np.percentile(train_scores, HIGH_PCT)),
        },
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "n_rows": int(len(df)),
        "neet_rate": float(y.mean()),
        "features_used": FEATURES,
        "cv_auc_logreg": [float(cv_logreg.mean()), float(cv_logreg.std())],
        "cv_auc_gbm": [float(cv_gbm.mean()), float(cv_gbm.std())],
        "holdout": holdout,
        "fairness_audit": fairness.reset_index().to_dict(orient="records"),
        "note": "Synthetic data: metrics show the model learns the relationships "
        "built into the dataset, not real-world predictive performance.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
