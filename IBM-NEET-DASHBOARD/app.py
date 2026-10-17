from pathlib import Path
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Get Your HEAD In The Game", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import db_utils
from model.inference import load_bundle, score

DATA_PATH = ROOT / "data" / "neet_dataset.csv"
FACTOR_COLUMNS = ["top_factor_1", "top_factor_2", "top_factor_3"]


@st.cache_resource
def get_model():
    return load_bundle()


def get_learners(data_source):
    if data_source == "Configured database":
        return db_utils.read_learners()
    return pd.read_csv(DATA_PATH)


def score_learners(learners):
    if learners.empty:
        raise ValueError("The selected data source contains no learners.")
    if "learner_id" not in learners.columns:
        raise ValueError("The selected data source must include a learner_id column.")

    predictions = score(learners, get_model())
    return learners.merge(
        predictions,
        on="learner_id",
        how="left",
        validate="one_to_one",
    )


st.sidebar.header("Data")
data_source = st.sidebar.radio(
    "Learner data source",
    ["Local synthetic dataset", "Configured database"],
    help=(
        "The local option uses data/neet_dataset.csv. The database option reads "
        "the configured NEET_DATA table using the DB_BACKEND environment setting."
    ),
)

try:
    learners = get_learners(data_source)
    df = score_learners(learners)
except Exception as error:
    st.error(f"Unable to load and score learners from {data_source.lower()}.")
    st.exception(error)
    st.stop()

st.sidebar.header("Filters")
selected_risk = st.sidebar.selectbox(
    "Risk level",
    ["All", "High", "Medium", "Low"],
)
filtered_df = (
    df if selected_risk == "All" else df[df["risk_level"] == selected_risk]
)

if filtered_df.empty:
    st.info("No learners match the selected risk-level filter.")
    st.stop()

st.title("🤜 Get Your HEAD In The Game 🤛")
st.subheader("By HEAD IBM Z for the IBM Z Datathon 2026")
st.markdown(
    "An early-warning dashboard that uses the trained NEET risk model to help "
    "identify where timely, human-led support may be useful."
)

if data_source == "Configured database":
    st.sidebar.caption(
        "Predictions are calculated from the current learner table. Saving them "
        "replaces the configured risk-score table."
    )
    confirm_save = st.sidebar.checkbox(
        "I understand this replaces all existing risk scores"
    )
    if st.sidebar.button("Save scores to database", disabled=not confirm_save):
        try:
            saved_count = db_utils.write_scores(df)
        except Exception as error:
            st.sidebar.error("Unable to save scores to the configured database.")
            st.sidebar.exception(error)
        else:
            st.sidebar.success(f"Saved {saved_count} learner scores.")

overview_tab, ranking_tab, profile_tab, risk_factors_tab, intervention_tab, ethics_tab = st.tabs(
    [
        "Overview",
        "Risk Ranking",
        "Individual Profile",
        "Top Risk Factors",
        "Interventions",
        "Ethics and Governance",
    ]
)

with overview_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learners", len(filtered_df))
    col2.metric("High Risk", int((filtered_df["risk_level"] == "High").sum()))
    col3.metric("Average Risk", f"{filtered_df['risk_score'].mean():.0%}")
    col4.metric(
        "Medium / High Risk",
        int(filtered_df["risk_level"].isin(["Medium", "High"]).sum()),
    )

    highest_risk = filtered_df.loc[filtered_df["risk_score"].idxmax()]
    st.info(
        f"Learner {highest_risk['learner_id']} currently has the highest predicted "
        f"risk ({highest_risk['risk_score']:.0%}). Predictions are advisory and "
        "must be reviewed by a person."
    )
    if data_source == "Local synthetic dataset":
        st.caption(
            "These predictions use synthetic demonstration data; they are not "
            "evidence about real learners."
        )

with ranking_tab:
    st.dataframe(
        filtered_df.sort_values("risk_score", ascending=False),
        width="stretch",
        hide_index=True,
    )

with profile_tab:
    selected_id = st.selectbox("Select learner", filtered_df["learner_id"].tolist())
    person = filtered_df.loc[filtered_df["learner_id"] == selected_id].iloc[0]
    profile_cols = st.columns(3)
    profile_cols[0].metric("Predicted risk", f"{person['risk_score']:.0%}")
    profile_cols[1].metric("Risk level", person["risk_level"])
    profile_cols[2].metric("Age", int(person["age"]))
    st.subheader("Model explanation")
    for column in FACTOR_COLUMNS:
        if person[column]:
            st.write(f"- {person[column]}")
    st.caption(
        "Explanations describe model associations, not causes or a full account "
        "of a learner's circumstances."
    )

with risk_factors_tab:
    st.subheader("Factors contributing to learner risk scores")
    factors = (
        filtered_df[FACTOR_COLUMNS]
        .melt(value_name="Factor")["Factor"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    if factors.empty:
        st.info("The selected learners have no positive risk-contributing factors.")
    else:
        st.bar_chart(factors.rename("Learners"))
    st.caption(
        "Factors are derived from each prediction's positive model contributions. "
        "They indicate association, not causation."
    )

with intervention_tab:
    st.subheader("Suggested support")
    selected_id = st.selectbox(
        "Select learner for support planning",
        filtered_df["learner_id"].tolist(),
        key="intervention_select",
    )
    person = filtered_df.loc[filtered_df["learner_id"] == selected_id].iloc[0]
    st.write(f"**Risk level:** {person['risk_level']}")
    st.write(f"**Model suggestion:** {person['suggested_intervention']}")
    st.info(
        "Suggestions are generated from the model's leading risk factor. A "
        "qualified support team should review the context and decide whether "
        "any action is appropriate."
    )

with ethics_tab:
    st.header("Responsible AI & Governance")
    st.subheader("Human-centred decision making")
    st.success(
        "Risk scores are advisory indicators only. All decisions must remain "
        "under human review and consider the learner's individual circumstances."
    )
    st.subheader("Explainability and fairness")
    st.info(
        "The model provides contributing factors, but explanations do not establish "
        "causation. Outputs should be routinely audited for demographic and "
        "geographical bias."
    )
    st.subheader("Privacy and purpose limitation")
    st.warning(
        "Use data only for appropriate early-support planning. Protect personal "
        "information, restrict access, and never use risk scores to penalise, "
        "discriminate against, or restrict opportunities."
    )
    st.markdown(
        """
        The objective is not prediction for its own sake. It is to support
        informed, timely, and proportionate help.
        """
    )
