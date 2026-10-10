"""Data loading and target creation. No modelling logic lives here."""
import pandas as pd

from src.config import DATASETS, PASS_THRESHOLD, TARGET_COL, GRADE_COLS, SCENARIOS


def load_raw(name: str = "por") -> pd.DataFrame:
    """Load a raw UCI dataset ('por' or 'mat').
    sep=None auto-detects ';' (original UCI files) or ',' (re-saved copies).
    """
    path = DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the CSV in data/raw/.")
    return pd.read_csv(path, sep=None, engine="python")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: 1 = fail (G3 < threshold), 0 = pass. Returns a copy."""
    df = df.copy()
    df[TARGET_COL] = (df["G3"] < PASS_THRESHOLD).astype(int)
    return df


def get_X_y(df: pd.DataFrame, scenario: str):
    """Return X, y for a scenario. Always drops G3 + target, and drops any
    grade column the scenario may not see (prevents leakage)."""
    allowed = SCENARIOS[scenario]
    blocked = [g for g in GRADE_COLS if g not in allowed]
    X = df.drop(columns=blocked + [TARGET_COL])
    y = df[TARGET_COL]
    return X, y