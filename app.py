
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Load the trained pipeline and expected feature columns
@st.cache_resource
def load_model():
    pipeline = joblib.load(
        BASE_DIR / "models" / "edupredict_pipeline.joblib"
    )
    columns = joblib.load(
        BASE_DIR / "models" / "feature_columns.joblib"
    )
    return pipeline, columns

# /////// pipipline.predict ici pour eviter les repetition op
    "Parental_Involvement": {
        "Low": 0, "Medium": 1, "High": 2
    },
    "Access_to_Resources": {
        "Low": 0, "Medium": 1, "High": 2
    },
    "Motivation_Level": {
        "Low": 0, "Medium": 1, "High": 2
    },
    "Family_Income": {
        "Low": 0, "Medium": 1, "High": 2
    },
    "Teacher_Quality": {
        "Low": 0, "Medium": 1, "High": 2
    },
    "Parental_Education_Level": {
        "High School": 0,
        "College": 1,
        "Postgraduate": 2
    },
    "Distance_from_Home": {
        "Near": 0,
        "Moderate": 1,
        "Far": 2
    }
}

nominal_cols = [
    "Gender",
    "School_Type",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Peer_Influence"
]


def preprocess_student(student, expected_columns):
    df = pd.DataFrame([student])

    # Ordinal encoding
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

        if df[col].isna().any():
            raise ValueError(
                f"Invalid category for {col}"
            )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=nominal_cols,
        drop_first=False,
        dtype=int
    )

    # Align with training columns
    df = df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    return df


st.title("🎓 EduPredict")
st.write("Estimate a student's exam score.")

try:
    pipeline, expected_columns = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


with st.form("student_form"):

    st.subheader("Study habits")

    col1, col2 = st.columns(2)

    with col1:
        hours = st.slider("Hours studied", 0, 50, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        previous = st.slider("Previous scores", 0, 100, 70)

    with col2:
        sleep = st.slider("Sleep hours", 0, 12, 7)
        tutoring = st.slider("Tutoring sessions", 0, 10, 1)
        activity = st.slider("Physical activity", 0, 10, 3)

    st.subheader("Student profile")

    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        school = st.selectbox(
            "School type", ["Public", "Private"]
        )

        motivation = st.selectbox(
            "Motivation level", ["Low", "Medium", "High"]
        )

        resources = st.selectbox(
            "Access to resources", ["Low", "Medium", "High"]
        )

        involvement = st.selectbox(
            "Parental involvement", ["Low", "Medium", "High"]
        )

        income = st.selectbox(
            "Family income", ["Low", "Medium", "High"]
        )

        teacher = st.selectbox(
            "Teacher quality", ["Low", "Medium", "High"]
        )

        education = st.selectbox(
            "Parental education",
            ["High School", "College", "Postgraduate"]
        )

    with col4:
        distance = st.selectbox(
            "Distance from home",
            ["Near", "Moderate", "Far"]
        )

        extracurricular = st.selectbox(
            "Extracurricular activities", ["Yes", "No"]
        )

        internet = st.selectbox(
            "Internet access", ["Yes", "No"]
        )

        disabilities = st.selectbox(
            "Learning disabilities", ["Yes", "No"]
        )

        peer = st.selectbox(
            "Peer influence",
            ["Positive", "Neutral", "Negative"]
        )

    st.subheader("Assessment settings")

    threshold = st.number_input(
        "School's minimum passing score",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    submitted = st.form_submit_button(
        "Predict exam score"
    )


if submitted:

    student = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": previous,
        "Sleep_Hours": sleep,
        "Tutoring_Sessions": tutoring,
        "Physical_Activity": activity,
        "Gender": gender,
        "School_Type": school,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access": internet,
        "Learning_Disabilities": disabilities,
        "Peer_Influence": peer,
        "Parental_Involvement": involvement,
        "Access_to_Resources": resources,
        "Motivation_Level": motivation,
        "Family_Income": income,
        "Teacher_Quality": teacher,
        "Parental_Education_Level": education,
        "Distance_from_Home": distance
    }

    try:
        student_df = preprocess_student(
            student, expected_columns
        )

        prediction = float(
            pipeline.predict(student_df)[0]
        )

        st.subheader("Prediction results")
        st.metric(
            "Estimated exam score",
            f"{prediction:.2f}"
        )

        if prediction < threshold:
            st.warning(
                "The estimated score is below the "
                "configured threshold. Additional "
                "academic support may be appropriate."
            )
        else:
            st.success(
                "The estimated score is at or above "
                "the configured threshold."
            )

        st.caption(
            "This is a statistical estimate, not a "
            "guarantee of an individual student's result."
        )

    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Invalid input: {exc}")