# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

def build_preprocessing_pipeline(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Use sparse_output if available, else fallback to sparse for older sklearn
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False) if "sparse_output" in OneHotEncoder().__init__.__code__.co_varnames else OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # --- fit ---
    X_proc = preprocessor.fit_transform(X)

    # --- save feature names for later use (SHAP, etc.) ---
    try:
        preprocessor.feature_names_ = preprocessor.get_feature_names_out()
    except AttributeError:
        # very old sklearn fallback
        preprocessor.feature_names_ = [f"f{i}" for i in range(X_proc.shape[1])]

    return preprocessor, X_proc
