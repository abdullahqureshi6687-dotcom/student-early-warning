"""
Train the final model on ALL Portuguese students and save it with joblib.

Run from the project root:
    python -m src.train_final

What gets saved (models/final_model.joblib) is a small "bundle" dict, so the
app needs nothing else:
- pipeline:   preprocessing + Logistic Regression (takes RAW student data)
- reference:  the training features, used as the "average student" when
              explaining a prediction
- risk_bands: probability cut-offs for Low / Medium / High, plus the fail rate
              actually observed in each band (from out-of-fold predictions,
              so these numbers are honest, not memorised)
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_predict

from src.config import MODELS_DIR
from src.data import add_target, get_X_y, load_raw
from src.evaluate import get_cv, make_estimator
from src.models import get_models

SCENARIO = "b_G1_only"          # chosen in Stage 4
MODEL_NAME = "Logistic Regression"
BAND_EDGES = [0.0, 0.3, 0.6, 1.0]   # assumption: Low < 0.3 <= Medium < 0.6 <= High
BAND_NAMES = ["Low", "Medium", "High"]


def risk_band(prob: float) -> str:
    """Map a fail probability to a risk band name."""
    for name, upper in zip(BAND_NAMES, BAND_EDGES[1:]):
        if prob < upper:
            return name
    return BAND_NAMES[-1]


def main():
    df = add_target(load_raw("por"))
    X, y = get_X_y(df, SCENARIO)
    model, scale, grid = get_models()[MODEL_NAME]

    # 1) Honest probabilities for every student (each from a model that never saw them)
    oof_prob = cross_val_predict(make_estimator(model, scale, grid, X), X, y,
                                 cv=get_cv(), method="predict_proba")[:, 1]
    bands = pd.Series([risk_band(p) for p in oof_prob])
    band_stats = (pd.DataFrame({"band": bands, "fail": y.values})
                  .groupby("band")["fail"].agg(students="count", fail_rate="mean")
                  .reindex(BAND_NAMES))
    print("Observed fail rate per risk band (out-of-fold):")
    print(band_stats.round(3), "\n")

    # 2) Final model: tune + fit on all students
    search = make_estimator(model, scale, grid, X).fit(X, y)
    print("Best params:", search.best_params_)

    bundle = {
        "pipeline": search.best_estimator_,
        "reference": X,
        "scenario": SCENARIO,
        "model_name": MODEL_NAME,
        "band_edges": BAND_EDGES,
        "band_names": BAND_NAMES,
        "band_stats": band_stats.to_dict(orient="index"),
        "sklearn_version": sklearn.__version__,   # must match when loading
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "final_model.joblib"
    joblib.dump(bundle, path)
    print(f"Saved -> {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()