import streamlit as st
import pandas as pd

factor_data = pd.DataFrame({
    "Factor": [
        "Attendance",
        "Mental Health",
        "Income",
        "Benefits",
        "Qualifications"
    ],
    "Importance": [
        42,
        25,
        15,
        12,
        6
    ]
})



st.set_page_config(
    page_title="Get Your HEAD In The Game",
    layout="wide"
)

df = pd.read_csv(
    "data/sample_predictions.csv"
)

st.sidebar.header("Filters")

selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + list(df["region"].unique())
)

filtered_df = df

region_risk = (
    filtered_df.groupby("region")["risk_score"]
    .mean()
    .reset_index()
)

top_region = region_risk.sort_values(
    by="risk_score",
    ascending=False
).iloc[0]

def risk_level(score):

    if score >= 0.8:
        return "🔴 High"

    elif score >= 0.5:
        return "🟠 Medium"

    else:
        return "🟢 Low"

df["Risk Level"] = (
    df["risk_score"]
    .apply(risk_level)
)

if selected_region != "All":
    filtered_df = df[
        df["region"] == selected_region
    ]

regions_requiring_support = len(
    region_risk[
        region_risk["risk_score"] >= 0.6
    ])

st.title("🤜 Get Your HEAD In The Game 🤛")

st.subheader(
    "By HEAD IBM Z for the IBM Z Datathon 2026"
)

st.markdown("""
Early warning system for identifying young people at risk of becoming NEET by identifying where support is needed earliest, so interventions can be offered before negative outcomes occur.
""")

overview_tab, ranking_tab, profile_tab, risk_factors_tab, intervention_tab, ethics_tab = st.tabs([
    "Overview",
    "Risk Ranking",
    "Individual Profile",
    "Top Risk Factors",
    "Interventions",
    "Ethics and Governance"
])

with overview_tab:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Population",
        len(filtered_df)
    )

    col2.metric(
        "High Risk Cases",
        len(filtered_df[filtered_df["risk_score"] > 0.8])
    )

    col3.metric(
        "Average Risk",
        f"{round(filtered_df['risk_score'].mean()*100)}%"
    )

    col4.metric(
        "Regions Requiring Support",
        regions_requiring_support
    )

    st.info(
        f"""
        Community-level analysis suggests that
        **{top_region['region']}** may benefit from
        additional youth engagement and support resources.
        """
    )

with ranking_tab:
    st.dataframe(
    filtered_df.sort_values(
        by="risk_score",
        ascending=False
    )
)

with profile_tab:
    selected_id = st.selectbox(
    "Select Individual",
    filtered_df["id"]
    )

    person = filtered_df[
    filtered_df["id"] == selected_id
    ]

    st.write(person)


with risk_factors_tab:
    st.subheader(
        "Top Risk Factors Driving Predicted NEET Outcomes"
    )

    st.write(
        """
        🔴 School attendance below 70%

        🟠 Poor mental health indicators

        🟠 Low household income

        🟠 Long-term benefit dependency

        🟢 Qualification attainment level
        """
    )

    st.bar_chart(
    factor_data.set_index("Factor") 
)

    st.success(
    """
    The model highlights factors associated
    with increased support needs but does not
    establish causation.
    """
    )  

with intervention_tab:
    st.subheader(
    "AI-Assisted Recommended Interventions"
    )

    st.success(
        """
        • Attendance mentoring

        • Mental health referral

        • Skills development programme

        • Employment readiness support
        """
    )

    selected_id = st.selectbox(
        "Select Individual for Support Planning",
        filtered_df["id"],
        key="intervention_select"
    )

    person = filtered_df[
        filtered_df["id"] == selected_id
    ].iloc[0]

    recommendations = []

    if person["attendance"] == "Poor":
        recommendations.append(
            "📚 Assign attendance mentor and school engagement support."
        )

    if person["mental_health"] == "Poor":
        recommendations.append(
            "🧠 Refer to mental wellbeing services and counselling support."
        )

    if person["risk_score"] > 0.8:
        recommendations.append(
            "🚨 Prioritise case review by youth support team."
        )

    st.subheader("Support Recommendations")

    for recommendation in recommendations:
        st.success(recommendation)

    st.subheader("Why these recommendations?")

    st.info(
        """
        Recommendations are generated using explainable
        risk indicators and are intended to support human
        decision-makers, not replace them.
        """
    )

with ethics_tab:
    st.header("Responsible AI & Governance")

    st.subheader("Human-Centred Decision Making")

    st.success(
        """
        Risk scores are advisory indicators only.
        All decisions remain under human review. This is not
        about removing human involvement but rather about providing 
        humans with the most informative data in order to utilise 
        a whole-of-government approach to help young people live full 
        and economically active lives.
        """
    )

    st.subheader("Explainable AI")

    st.info(
        """
        Every prediction is accompanied by the
        contributing factors that influenced the
        risk assessment.
        """
    )

    st.subheader("Fairness Monitoring")

    st.info(
        """
        Model outputs should be routinely audited
        for demographic and geographical bias.
        """
    )

    st.subheader("Privacy & Data Protection")

    st.info(
        """
        Personally identifiable information is
        minimised and protected throughout the
        analytical process.
        """
    )

    st.subheader("Purpose Limitation")

    st.warning(
        """
        This platform is designed exclusively to
        support early intervention and improved
        outcomes for young people.

        It must not be used to penalise,
        discriminate against, or restrict
        opportunities.
        """
    )

    st.subheader("AI for Good Impact")

    st.markdown("""
    ### Our Commitment

    This platform exists to:

    - Identify support needs earlier
    - Improve educational outcomes
    - Reduce social exclusion
    - Help government allocate resources effectively
    - Strengthen collaboration between public, private and third-sector organisations

    The objective is not prediction.

    The objective is prevention.
    """
    )



