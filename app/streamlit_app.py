"""
Student Early-Warning System: Streamlit app.

Run locally from the project root:   streamlit run app/streamlit_app.py
Deployed on Streamlit Community Cloud with main file path: app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Paths: the app lives in app/, src/ and models/ are one level up ----------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.explain import explain_student  # noqa: E402  (needs the path fix above)

st.set_page_config(page_title="Student Early-Warning System", page_icon="🎓", layout="wide")


@st.cache_resource  # load the model ONCE, not on every click
def load_bundle():
    return joblib.load(ROOT / "models" / "final_model.joblib")


bundle = load_bundle()
PIPE = bundle["pipeline"]
REFERENCE = bundle["reference"]          # training features = "average student"
BAND_EDGES = bundle["band_edges"]
BAND_NAMES = bundle["band_names"]
BAND_STATS = bundle["band_stats"]
FEATURES = list(REFERENCE.columns)       # exact column order the model expects

# ---------- Human-friendly labels (UCI codebook) ----------
LABELS = {
    "G1": "First-period grade G1 (0–20)",
    "failures": "Past class failures",
    "school": "School (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
    "higher": "Wants to take higher education",
    "age": "Age",
    "studytime": "Weekly study time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h)",
    "absences": "Number of school absences",
    "sex": "Sex", "address": "Home address (U = urban, R = rural)",
    "famsize": "Family size (LE3 = ≤3, GT3 = >3)",
    "Pstatus": "Parents' status (T = together, A = apart)",
    "Medu": "Mother's education (0 none – 4 higher)",
    "Fedu": "Father's education (0 none – 4 higher)",
    "Mjob": "Mother's job", "Fjob": "Father's job",
    "reason": "Reason for choosing this school", "guardian": "Guardian",
    "traveltime": "Travel time (1: <15min – 4: >1h)",
    "schoolsup": "Extra school support", "famsup": "Family educational support",
    "paid": "Extra paid classes", "activities": "Extra-curricular activities",
    "nursery": "Attended nursery school", "internet": "Internet at home",
    "romantic": "In a romantic relationship",
    "famrel": "Family relationship quality (1–5)", "freetime": "Free time (1–5)",
    "goout": "Going out with friends (1–5)", "Dalc": "Workday alcohol use (1–5)",
    "Walc": "Weekend alcohol use (1–5)", "health": "Health status (1–5)",
}
KEY_FEATURES = ["G1", "failures", "school", "higher", "age", "studytime", "absences"]
FIXED_RANGES = {"G1": (0, 20)}  # assumption: the full 0–20 scale, not just observed values
BAND_ICONS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}


def make_input(col: str):
    """Build a Streamlit input from the training data: choices, range and default."""
    series = REFERENCE[col]
    label = LABELS.get(col, col)
    if not pd.api.types.is_numeric_dtype(series):
        choices = sorted(series.unique().tolist())
        default = series.mode()[0]                       # most common value
        if len(choices) == 2:
            return st.radio(label, choices, index=choices.index(default), horizontal=True)
        return st.selectbox(label, choices, index=choices.index(default))
    lo, hi = FIXED_RANGES.get(col, (int(series.min()), int(series.max())))
    return st.slider(label, lo, hi, int(series.median()))


def risk_band(prob: float) -> str:
    for name, upper in zip(BAND_NAMES, BAND_EDGES[1:]):
        if prob < upper:
            return name
    return BAND_NAMES[-1]


# ---------- Layout ----------
st.title("🎓 Student Early-Warning System")
st.markdown(
    "Estimates the risk that a student **fails the final grade (G3 < 10)** using "
    "background information and the **first-period grade (G1)**, early enough in "
    "the year to intervene. Trained on the UCI Student Performance dataset "
    "(Portuguese course, 649 students)."
)

left, right = st.columns(2, gap="large")

with left:
    with st.form("student"):
        st.subheader("Key information")
        values = {col: make_input(col) for col in KEY_FEATURES}
        with st.expander("More details (optional, pre-filled with typical values)"):
            for col in FEATURES:
                if col not in KEY_FEATURES:
                    values[col] = make_input(col)
        submitted = st.form_submit_button("Predict risk", type="primary",
                                          use_container_width=True)

with right:
    if not submitted:
        st.info("Fill in the details and click **Predict risk**.")
    else:
        student = pd.DataFrame([values])[FEATURES].astype(REFERENCE.dtypes.to_dict())
        prob = float(PIPE.predict_proba(student)[0, 1])
        band = risk_band(prob)
        stats = BAND_STATS[band]

        st.subheader(f"{BAND_ICONS[band]} {band} risk")
        st.metric("Model fail-risk score", f"{prob:.0%}")
        st.caption(f"In cross-validation, {stats['fail_rate']:.0%} of the "
                   f"{stats['students']} students in the {band} band actually failed.")

        st.markdown("#### Top 3 reasons")
        st.caption("What moved this score most, compared with an average student:")
        reasons = explain_student(PIPE, student, REFERENCE, top_n=3)
        for _, r in reasons.iterrows():
            arrow = "⬆️ raises" if r["effect"] == "raises risk" else "⬇️ lowers"
            st.markdown(f"- **{LABELS.get(r['feature'], r['feature'])}** = "
                        f"`{r['value']}` → {arrow} risk")

    st.markdown("---")
    st.caption("Limitations: trained on ~650 students from two Portuguese schools. "
               "Risk factors are associations, not causes. Use this to decide who to "
               "check in with, never as a label for a student.")