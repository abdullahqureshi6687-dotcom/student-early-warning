"""
Explainability helpers for the final model (Logistic Regression in a Pipeline).

Three views, from global to personal:
1. coefficient_table()          -> what the model learned (direction + strength)
2. cv_permutation_importance()  -> how much each ORIGINAL feature matters for
                                   performance on unseen students
3. explain_student()            -> top reasons for ONE student's risk (used by the app)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src.config import N_SPLITS, RANDOM_STATE
from src.evaluate import get_cv


def original_feature(encoded_name: str, original_cols) -> str:
    """Map an encoded column name back to its raw feature.

    e.g. "Mjob_teacher" -> "Mjob", "higher_yes" -> "higher", "failures" -> "failures".
    Longest match first, so a name like "Fjob_at_home" can't be confused.
    """
    for col in sorted(original_cols, key=len, reverse=True):
        if encoded_name == col or encoded_name.startswith(col + "_"):
            return col
    return encoded_name


def coefficient_table(pipe) -> pd.DataFrame:
    """Logistic Regression coefficients with readable names.

    Numeric inputs are standardised, so a coefficient = change in log-odds of
    failing for a +1 standard deviation change. For one-hot columns it is the
    change for having that category vs the reference category.
    odds_ratio = exp(coef): > 1 raises fail odds, < 1 lowers them.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coef": coefs})
    table["odds_ratio"] = np.exp(table["coef"])
    table["direction"] = np.where(table["coef"] > 0, "raises risk", "lowers risk")
    return table.reindex(table["coef"].abs().sort_values(ascending=False).index)


def cv_permutation_importance(pipe, X, y, n_repeats: int = 10,
                              scoring: str = "average_precision") -> pd.DataFrame:
    """Permutation importance measured on held-out folds only.

    For each CV fold: fit on the training part, then shuffle one ORIGINAL
    column at a time in the test part and measure how much PR-AUC drops.
    A big drop = the model really relies on that feature.
    Computing it on unseen data avoids rewarding features the model merely
    memorised.
    """
    drops = []
    for train_idx, test_idx in get_cv(N_SPLITS).split(X, y):
        model = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            model, X.iloc[test_idx], y.iloc[test_idx], scoring=scoring,
            n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
        drops.append(result.importances_mean)

    drops = np.array(drops)  # shape: (folds, features)
    return (pd.DataFrame({"feature": X.columns,
                          "importance": drops.mean(axis=0),
                          "std": drops.std(axis=0)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def explain_student(pipe, student: pd.DataFrame, X_reference: pd.DataFrame,
                    top_n: int = 3) -> pd.DataFrame:
    """Top reasons pushing ONE student's fail risk up or down.

    For a linear model, each encoded column contributes
        coef * (student value - average student value)
    to the log-odds of failing (this is exactly what SHAP computes for a
    linear model). We sum one-hot columns back into their original feature,
    so "Mjob_teacher" and "Mjob_other" both count as "Mjob".
    Positive contribution = raises risk, negative = lowers risk.
    """
    prep = pipe.named_steps["prep"]
    coefs = pipe.named_steps["model"].coef_[0]
    names = prep.get_feature_names_out()

    x_student = prep.transform(student)[0]
    x_average = prep.transform(X_reference).mean(axis=0)
    contrib = coefs * (x_student - x_average)

    per_feature = (pd.Series(contrib, index=[original_feature(n, student.columns) for n in names])
                   .groupby(level=0).sum())
    top = per_feature.reindex(per_feature.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame({
        "feature": top.index,
        "value": [student.iloc[0][f] for f in top.index],
        "contribution": top.values,
        "effect": np.where(top.values > 0, "raises risk", "lowers risk"),
    })