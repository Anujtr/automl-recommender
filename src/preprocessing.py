# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

def build_preprocessing_pipeline(X: pd.DataFrame, use_polynomial=False, use_interactions=False):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe_steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ]
    if use_polynomial or use_interactions:
        degree = 2
        interaction_only = use_interactions and not use_polynomial
        num_pipe_steps.append(
            ("poly", PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False))
        )

    num_pipe = Pipeline(num_pipe_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False) if "sparse_output" in OneHotEncoder().__init__.__code__.co_varnames else OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_proc = preprocessor.fit_transform(X)

    try:
        preprocessor.feature_names_ = preprocessor.get_feature_names_out()
    except AttributeError:
        preprocessor.feature_names_ = [f"f{i}" for i in range(X_proc.shape[1])]

    return preprocessor, X_proc
