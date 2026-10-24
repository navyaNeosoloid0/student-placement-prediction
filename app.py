
"""Student Placement Prediction System using a trained ML model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
/* Main background and typography */
.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #101c30 100%);
    color: #e5edf7;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
h1 {
    color: #f1f5f9 !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #dce8f5 !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #263449;
}

.stTabs [data-baseweb="tab"] {
    background: #172438;
    border-radius: 10px 10px 0 0;
    padding: 12px 20px;
    color: #cbd5e1;
}

.stTabs [aria-selected="true"] {
    background: #15364a !important;
    color: #55d6be !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #172438;
    border: 1px solid #293b53;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.16);
}

[data-testid="stMetricLabel"] {
    color: #a9bbcf !important;
}

[data-testid="stMetricValue"] {
    color: #55d6be !important;
    font-weight: 750;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0f9f91, #18b6a5);
    color: #061521;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #55d6be;
    color: #061521;
    border: none;
    transform: translateY(-1px);
}

/* Input fields */
.stSlider, [data-testid="stNumberInput"],
[data-testid="stSelectbox"] {
    background: #111e30;
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    background-color: #172438;
    border-color: #354860;
    border-radius: 9px;
}

/* Separators and progress bar */
hr {
    border-color: #293b53 !important;
}

.stProgress > div > div > div {
    background: #55d6be;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Images and data tables */
[data-testid="stImage"] img {
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293b53;
    border-radius: 12px;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Fix visibility of labels and input text in dark theme */
.stApp,
.stApp p,
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp .stMarkdown,
.stApp .stMarkdown p {
    color: #E5EDF7 !important;
}

/* Slider labels and values */
.stSlider label,
.stSlider [data-testid="stThumbValue"] {
    color: #F1F5F9 !important;
}

/* Number input and dropdown text */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] span {
    color: #172033 !important;
}

/* Improve tab text contrast */
.stTabs [data-baseweb="tab"] p {
    color: #E5EDF7 !important;
}

.stTabs [aria-selected="true"] p {
    color: #55D6BE !important;
}

/* Keep muted captions readable */
.stApp [data-testid="stCaptionContainer"] p {
    color: #AAB8CA !important;
}

/* Input field labels */
.stApp [data-testid="stWidgetLabel"] {
    color: #E5EDF7 !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/meta.json")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with META_PATH.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    return model, meta


def generate_recommendations(profile):
    """Generate rule-based suggestions from a student's profile."""
    recommendations = []

    if profile["CGPA"] < 7.0:
        recommendations.append(
            "Academic performance: Strengthen core subjects and aim "
            "to improve your CGPA where possible."
        )

    if profile["Internships"] == 0:
        recommendations.append(
            "Experience: Look for internships, virtual internships, "
            "or practical experience relevant to your target role."
        )
    elif profile["Internships"] == 1:
        recommendations.append(
            "Experience: Build on your existing internship experience "
            "with practical work related to your target job."
        )

    if profile["Projects"] < 2:
        recommendations.append(
            "Projects: Build at least two practical, job-relevant "
            "projects and document them on GitHub."
        )
    elif profile["Projects"] < 4:
        recommendations.append(
            "Projects: Improve your existing projects with clear "
            "documentation, testing, and working demonstrations."
        )

    if profile["Workshops/Certifications"] < 2:
        recommendations.append(
            "Certifications: Consider relevant courses or certifications "
            "that help demonstrate job-ready skills."
        )

    if profile["AptitudeTestScore"] < 60:
        recommendations.append(
            "Aptitude: Practise quantitative aptitude, logical reasoning, "
            "and verbal ability regularly."
        )
    elif profile["AptitudeTestScore"] < 75:
        recommendations.append(
            "Aptitude: Improve speed and accuracy through timed practice "
            "tests and review your mistakes."
        )

    if profile["SoftSkillsRating"] < 3.0:
        recommendations.append(
            "Soft skills: Practise communication, presentations, "
            "teamwork, and mock interviews."
        )
    elif profile["SoftSkillsRating"] < 4.0:
        recommendations.append(
            "Soft skills: Continue improving interview answers, "
            "professional communication, and confidence."
        )

    if profile["ExtracurricularActivities"] == "No":
        recommendations.append(
            "Teamwork: Consider participating in clubs, hackathons, "
            "volunteering, or group activities to practise collaboration."
        )

    if profile["PlacementTraining"] == "No":
        recommendations.append(
            "Placement preparation: Consider mock interviews, resume "
            "reviews, aptitude sessions, and coding practice."
        )

    if profile["SSC_Marks"] < 70:
        recommendations.append(
            "SSC academics: Review your fundamentals and check the "
            "academic eligibility criteria for your target employers."
        )

    if profile["HSC_Marks"] < 70:
        recommendations.append(
            "HSC academics: Check employer-specific academic criteria "
            "and highlight your current technical strengths."
        )

    if not recommendations:
        recommendations.append(
            "Keep building job-relevant skills, practising interviews, "
            "and improving your projects. Continue applying to suitable "
            "roles and reviewing employer requirements."
        )

    return recommendations


model, meta = load_model()

st.title("🎓 Student Placement Prediction System")
st.write(
    "Enter your academic and extracurricular details to estimate "
    "your placement outcome."
)

metrics = meta.get("metrics", {})
st.caption(
    f"Model: {meta.get('best_model', 'Trained model')} | "
    f"F1: {metrics.get('F1', 0):.2%} | "
    f"Accuracy: {metrics.get('Accuracy', 0):.2%} | "
    f"ROC-AUC: {metrics.get('ROC-AUC', 0):.2%}"
)

tab_predict, tab_insights = st.tabs(
    ["🔮 Predict", "📊 Model Insights"]
)

with tab_predict:
    st.subheader("Student Profile")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        internships = st.number_input(
            "Number of Internships", 0, 20, 1
        )
        projects = st.number_input(
            "Number of Projects", 0, 30, 2
        )
        workshops = st.number_input(
            "Workshops / Certifications", 0, 30, 2
        )
        aptitude = st.slider(
            "Aptitude Test Score", 0, 100, 65
        )

    with col2:
        soft_skills = st.slider(
            "Soft Skills Rating", 0.0, 5.0, 3.5, 0.1
        )
        extracurricular = st.selectbox(
            "Extracurricular Activities", ["Yes", "No"]
        )
        placement_training = st.selectbox(
            "Placement Training", ["Yes", "No"]
        )
        ssc_marks = st.slider(
            "SSC Marks (%)", 0.0, 100.0, 75.0, 1.0
        )
        hsc_marks = st.slider(
            "HSC Marks (%)", 0.0, 100.0, 75.0, 1.0
        )

    if st.button("Predict Placement", type="primary"):
        student = pd.DataFrame([{
            "CGPA": cgpa,
            "Internships": internships,
            "Projects": projects,
            "Workshops/Certifications": workshops,
            "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft_skills,
            "ExtracurricularActivities": extracurricular,
            "PlacementTraining": placement_training,
            "SSC_Marks": ssc_marks,
            "HSC_Marks": hsc_marks,
        }])

        try:
            probabilities = model.predict_proba(student)[0]
            classes = list(model.classes_)

            # The training target is encoded as:
            # 0 = NotPlaced, 1 = Placed
            placed_index = classes.index(1)
            probability = float(probabilities[placed_index])

            prediction_value = model.predict(student)[0]
            prediction = (
                "Placed" if prediction_value == 1 else "Not Placed"
            )

            st.subheader("Prediction Result")
            st.metric(
                "Estimated placement probability",
                f"{probability:.1%}",
            )
            st.progress(probability)

            if prediction == "Placed":
                st.success("The model predicts: Placed")
            else:
                st.warning("The model predicts: Not Placed")

            st.info(
                "This is a machine-learning estimate based on the "
                "training dataset, not a guarantee of employment."
            )

            # Use the actual inputs to personalize recommendations.
            profile = {
                "CGPA": cgpa,
                "Internships": internships,
                "Projects": projects,
                "Workshops/Certifications": workshops,
                "AptitudeTestScore": aptitude,
                "SoftSkillsRating": soft_skills,
                "ExtracurricularActivities": extracurricular,
                "PlacementTraining": placement_training,
                "SSC_Marks": ssc_marks,
                "HSC_Marks": hsc_marks,
            }

            recommendations = generate_recommendations(profile)

            st.subheader("Suggestions for Improvement")
            st.write(
                "These suggestions are based on the details you entered, "
                "not on a guaranteed explanation of the model prediction."
            )

            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.write(
                "Check that the saved model and input columns match "
                "the training pipeline."
            )


with tab_insights:
    st.header("📊 Model Performance Dashboard")

    # 1. Selected model metrics
    metrics = meta.get("metrics", {})

    st.subheader(f"Selected Model: {meta.get('best_model', 'Trained Model')}")
    st.caption(
        "Metrics are based on the group-aware hold-out test set, "
        "except for cross-validation metrics."
    )

    metric_cols = st.columns(5)

    metric_cards = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1 Score", "F1"),
        ("ROC-AUC", "ROC-AUC"),
    ]

    for column, (label, key) in zip(metric_cols, metric_cards):
        value = metrics.get(key)
        column.metric(
            label,
            f"{value:.2%}" if value is not None else "N/A",
        )

    cv_cols = st.columns(2)
    cv_mean = metrics.get("CV F1 mean")
    cv_std = metrics.get("CV F1 std")

    cv_cols[0].metric(
        "Cross-validation F1 mean",
        f"{cv_mean:.2%}" if cv_mean is not None else "N/A",
    )
    cv_cols[1].metric(
        "Cross-validation F1 std",
        f"{cv_std:.3f}" if cv_std is not None else "N/A",
    )



    st.info(
        f"The selected model achieved "
        f"{metrics.get('Accuracy', 0):.2%} hold-out accuracy, "
        f"{metrics.get('F1', 0):.2%} F1 score, and "
        f"{metrics.get('ROC-AUC', 0):.2%} ROC-AUC on this dataset. "
        "These results do not establish real-world placement accuracy."
    )



    st.divider()

    # 2. Compare models
    st.subheader("Model Comparison")

    comparison_path = Path("reports/model_comparison.csv")

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

        if {"Model", "Accuracy", "F1"}.issubset(comparison_df.columns):
            chart_df = comparison_df.set_index("Model")[
                ["Accuracy", "F1"]
            ]
            st.write("Accuracy vs. F1 Score")
            st.bar_chart(chart_df, use_container_width=True)

        if {"Model", "CV F1 mean"}.issubset(comparison_df.columns):
            cv_chart = comparison_df.set_index("Model")[["CV F1 mean"]]
            st.write("Cross-validation F1 Mean")
            st.bar_chart(cv_chart, use_container_width=True)

        if {"Model", "F1", "CV F1 mean"}.issubset(comparison_df.columns):
            best_holdout = comparison_df.loc[
                comparison_df["F1"].idxmax()
            ]
            best_cv = comparison_df.loc[
                comparison_df["CV F1 mean"].idxmax()
            ]

            st.markdown(
                f"- **Highest hold-out F1:** {best_holdout['Model']} "
                f"({best_holdout['F1']:.2%})"
            )
            st.markdown(
                f"- **Highest cross-validation F1 mean:** "
                f"{best_cv['Model']} ({best_cv['CV F1 mean']:.2%})"
            )

            st.caption(
                "These are different evaluation procedures. For stricter "
                "model selection, use cross-validation to choose the model "
                "and reserve the hold-out test set for final evaluation."
            )
    else:
        st.warning("Model comparison report was not found.")

    st.divider()

    # 3. Feature importance
    st.subheader("Feature Importance")
    st.write(
        "Permutation importance estimates how much model performance "
        "changes when a feature is shuffled. It measures model reliance, "
        "not whether a feature causes placement."
    )

    importance_path = Path("reports/feature_importance.png")

    if importance_path.exists():
        st.image(
            str(importance_path),
            caption="Permutation Feature Importance — Test Set",
            use_container_width=True,
        )
    else:
        st.warning("Feature importance chart was not found.")

    st.divider()

    # 4. Confusion matrix
    st.subheader("Confusion Matrix")
    st.write(
        "The confusion matrix shows how the model classified the "
        "students in the hold-out test set."
    )

    confusion_path = Path("reports/confusion_matrix.png")

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption="Hold-out Test Set Confusion Matrix",
            use_container_width=True,
        )

        st.markdown(
            """
            - **True positive:** A student labeled Placed was predicted as Placed.
            - **True negative:** A student labeled NotPlaced was predicted as NotPlaced.
            - **False positive:** A student labeled NotPlaced was predicted as Placed.
            - **False negative:** A student labeled Placed was predicted as NotPlaced.
            """
        )
    else:
        st.warning("Confusion matrix image was not found.")

    st.divider()

    # 5. Exploratory data analysis
    st.subheader("Dataset Exploration")

    for filename, caption in [
        ("eda.png", "Exploratory Data Analysis"),
        ("correlation.png", "Feature Correlations"),
    ]:
        image_path = Path("reports") / filename

        if image_path.exists():
            st.markdown(f"**{caption}**")
            st.image(
                str(image_path),
                use_container_width=True,
            )
        else:
            st.warning(f"{caption} chart was not found.")

    st.divider()

    # 6. Dataset and evaluation notes
    st.subheader("Dataset and Evaluation Notes")

    st.write(f"**Dataset:** `{meta.get('data_source', 'Not specified')}`")
    st.write(
        f"**Hold-out split:** "
        f"`{meta.get('split_strategy', 'Not specified')}`"
    )
    st.write(
        f"**Cross-validation:** "
        f"`{meta.get('cross_validation_strategy', 'Not specified')}`"
    )

    st.write(
        f"**Duplicate profiles retained:** "
        f"{meta.get('duplicate_profiles_kept', 'Not specified')}"
    )

    st.warning(
        meta.get(
            "data_note",
            "Dataset provenance and independent real-world predictive "
            "validity have not been established.",
        )
    )

st.caption(
    "Educational project. Predictions reflect patterns in the supplied "
    "dataset and may not generalize to real-world placement outcomes."
)
