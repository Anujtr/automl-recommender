# src/preprocessing.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessing_pipeline(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # --- transformers ---
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler())
    ])

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", onehot)
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numerical_cols),
            ("cat", categorical_tf, categorical_cols)
        ]
    )
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # 🔑  Store feature names for SHAP / downstream use
    cat_names = onehot.get_feature_names_out(categorical_cols)
    preprocessor.feature_names_ = numerical_cols + cat_names.tolist()

    return preprocessor, X_processed
