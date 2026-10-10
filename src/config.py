"""
Central configuration: paths, constants and experiment scenarios.
Every notebook and script imports from here, so each value is defined ONCE.
"""
from pathlib import Path

# ---------- Paths (work no matter where you run the code from) ----------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

DATASETS = {
    "por": DATA_RAW_DIR / "student-por.csv",  # Portuguese (training)
    "mat": DATA_RAW_DIR / "student-mat.csv",  # Math (Stage 6 test)
}

# ---------- Reproducibility ----------
RANDOM_STATE = 42
N_SPLITS = 5  # stratified K-fold

# ---------- Target definition ----------
PASS_THRESHOLD = 10        # G3 >= 10 -> pass
TARGET_COL = "fail"        # 1 = FAIL (class we want to catch), 0 = pass
GRADE_COLS = ["G1", "G2", "G3"]

# ---------- Experiment scenarios (Stage 4) ----------
# Which grade columns each scenario is ALLOWED to see. G3 is never a feature.
SCENARIOS = {
    "a_no_grades": [],          # start of year: true early warning
    "b_G1_only": ["G1"],        # after 1st period
    "c_G1_G2": ["G1", "G2"],    # after 2nd period: late warning
}