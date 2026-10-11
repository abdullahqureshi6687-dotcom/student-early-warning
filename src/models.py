"""
Model definitions + small hyperparameter grids.

Each entry: name -> (estimator, scale_numeric, param_grid)
- scale_numeric: linear models need scaled inputs, tree models don't.
- param_grid: kept SMALL on purpose ("light tuning"). Keys use the
  "model__" prefix because the estimator sits inside a Pipeline step
  named "model".
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def get_models():
    return {
        # Always predicts the majority class (Pass). The bar every model must beat.
        "Baseline (majority)": (
            DummyClassifier(strategy="most_frequent"),
            False,
            None,
        ),
        # Linear, fast, interpretable. C = inverse regularisation strength.
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE),
            True,
            {"model__C": [0.01, 0.1, 1.0]},
        ),
        # Many decision trees voting. Captures non-linear patterns.
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            False,
            {"model__max_depth": [4, 8, None],
             "model__min_samples_leaf": [1, 5]},
        ),
        # Trees built one after another, each fixing the previous one's errors.
        # sklearn's histogram version: fast and supports class_weight.
        "Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced",
                                           random_state=RANDOM_STATE),
            False,
            {"model__learning_rate": [0.05, 0.1],
             "model__max_depth": [3, None]},
        ),
    }