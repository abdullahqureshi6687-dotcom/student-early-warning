"""
Evaluation: stratified cross-validation with (nested) light tuning.

Why nested CV? If we tune hyperparameters and report the score on the SAME
folds, the score is optimistic (we picked the settings that happened to win
on those folds). Nested CV tunes inside each training split (inner 3-fold)
and scores on an untouched outer fold, so the reported number stays honest.
"""
import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import N_SPLITS, RANDOM_STATE, SCENARIOS
from src.data import get_X_y
from src.models import get_models
from src.preprocessing import build_pipeline

# All metrics are computed for the positive class: fail = 1
SCORING = {
    "fail_recall": "recall",
    "fail_precision": make_scorer(precision_score, zero_division=0),
    "fail_f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
TUNING_METRIC = "average_precision"  # PR-AUC: robust choice for imbalanced data


def get_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Stratified folds keep the ~15% fail ratio in every fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def make_estimator(model, scale_numeric, param_grid, X):
    """Pipeline, wrapped in a GridSearchCV when there is something to tune."""
    pipe = build_pipeline(model, X, scale_numeric=scale_numeric)
    if not param_grid:
        return pipe
    return GridSearchCV(pipe, param_grid, scoring=TUNING_METRIC,
                        cv=get_cv(3), n_jobs=-1)


def evaluate_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run every model on every scenario. Returns one row per (scenario, model)."""
    rows = []
    for scenario in SCENARIOS:
        X, y = get_X_y(df, scenario)
        for name, (model, scale, grid) in get_models().items():
            est = make_estimator(model, scale, grid, X)
            res = cross_validate(est, X, y, cv=get_cv(), scoring=SCORING)
            row = {"scenario": scenario, "model": name}
            for metric in SCORING:
                row[metric] = res[f"test_{metric}"].mean()
                row[f"{metric}_std"] = res[f"test_{metric}"].std()
            rows.append(row)
            print(f"done: {scenario:12} | {name}")
    return pd.DataFrame(rows)