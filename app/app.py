"""
Student Early-Warning System: Gradio app.

Run locally from the project root:   python app/app.py
On Hugging Face Spaces this file sits at the Space root as app.py.
"""
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

# ---------- Paths: work both locally (repo/app/app.py) and on HF (root/app.py) ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT))

from src.explain import explain_student  # noqa: E402  (needs the path fix above)

# ---------- Load the saved bundle (trained by: python -m src.train_final) ----------
bundle = joblib.load(ROOT / "models" / "final_model.joblib")
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


def make_input(col: str):
    """Build a Gradio input from the training data: choices, range and default."""
    series = REFERENCE[col]
    label = LABELS.get(col, col)
    if series.dtype == object or str(series.dtype).startswith(("string", "str")):
        choices = sorted(series.unique().tolist())
        default = series.mode()[0]                       # most common value
        if len(choices) == 2:
            return gr.Radio(choices, value=default, label=label)
        return gr.Dropdown(choices, value=default, label=label)
    lo, hi = FIXED_RANGES.get(col, (int(series.min()), int(series.max())))
    return gr.Slider(lo, hi, value=int(series.median()), step=1, label=label)


def risk_band(prob: float) -> str:
    for name, upper in zip(BAND_NAMES, BAND_EDGES[1:]):
        if prob < upper:
            return name
    return BAND_NAMES[-1]


def predict(*values):
    """Inputs arrive in FEATURES order -> probability, band and top 3 reasons."""
    student = pd.DataFrame([dict(zip(FEATURES, values))])[FEATURES]
    # Match training dtypes (sliders return floats; the model saw ints)
    student = student.astype(REFERENCE.dtypes.to_dict())

    prob = float(PIPE.predict_proba(student)[0, 1])
    band = risk_band(prob)
    stats = BAND_STATS[band]
    icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[band]

    reasons = explain_student(PIPE, student, REFERENCE, top_n=3)
    reason_lines = []
    for _, r in reasons.iterrows():
        arrow = "⬆️ raises" if r["effect"] == "raises risk" else "⬇️ lowers"
        reason_lines.append(f"- **{LABELS.get(r['feature'], r['feature'])}** = "
                            f"`{r['value']}` → {arrow} risk")

    return (
        f"## {icon} {band} risk\n"
        f"**Model fail-risk score: {prob:.0%}**\n\n"
        f"In cross-validation, {stats['fail_rate']:.0%} of the "
        f"{stats['students']} students in the *{band}* band actually failed.\n\n"
        f"### Top 3 reasons\n"
        f"*What moved this score most, compared with an average student:*\n\n"
        + "\n".join(reason_lines)
    )


# ---------- Layout ----------
with gr.Blocks(title="Student Early-Warning System") as demo:
    gr.Markdown(
        "# 🎓 Student Early-Warning System\n"
        "Estimates the risk that a student **fails the final grade (G3 < 10)** "
        "using background information and the **first-period grade (G1)**, early "
        "enough in the year to intervene. Trained on the UCI Student Performance "
        "dataset (Portuguese course, 649 students)."
    )
    inputs = {}
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Key information")
            for col in KEY_FEATURES:
                inputs[col] = make_input(col)
            with gr.Accordion("More details (optional, pre-filled with typical values)",
                              open=False):
                for col in FEATURES:
                    if col not in KEY_FEATURES:
                        inputs[col] = make_input(col)
            button = gr.Button("Predict risk", variant="primary")
        with gr.Column():
            output = gr.Markdown("Fill in the details and click **Predict risk**.")
            gr.Markdown(
                "*Limitations: trained on ~650 students from two Portuguese schools. "
                "Risk factors are associations, not causes. Use this to decide who to "
                "check in with, never as a label for a student.*"
            )

    button.click(predict, inputs=[inputs[c] for c in FEATURES], outputs=output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())