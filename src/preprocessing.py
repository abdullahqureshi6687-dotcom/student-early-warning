"""
Preprocessing: turn raw student features into model-ready numbers.

Everything is wrapped in a scikit-learn Pipeline so that the scaler and the
encoder are fitted ONLY on training data (inside each CV fold). This is what
prevents data leakage from the test set into training.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X: pd.DataFrame):
    """Split columns into numeric and categorical lists, based on dtype."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Create a ColumnTransformer for the columns present in X.

    - Numeric columns -> StandardScaler (mean 0, std 1). Linear models need
      this; tree models don't, so scaling can be switched off.
    - Categorical columns -> OneHotEncoder. drop="if_binary" turns yes/no
      columns into ONE 0/1 column instead of two redundant ones.
      handle_unknown="ignore" means a category never seen in training
      (e.g. in the Math dataset) becomes all zeros instead of crashing.
    """
    num_cols, cat_cols = get_feature_types(X)
    numeric_step = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, num_cols),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,  # keep readable names like "school_MS"
    )


def build_pipeline(model, X: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Chain preprocessing + model into one object with .fit() / .predict()."""
    return Pipeline([
        ("prep", build_preprocessor(X, scale_numeric=scale_numeric)),
        ("model", model),
    ])