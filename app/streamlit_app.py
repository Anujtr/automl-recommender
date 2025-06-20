import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import build_preprocessing_pipeline
from model_selector import evaluate_models
from tuner import tune_multiple_models
from utils import get_feature_names
from explainer import explain_model
from evaluator import (
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
    save_model,
)
from config import TUNING_CONFIGS

st.set_page_config(page_title="AutoML Model Recommender", layout="wide")
st.title("🎯 AutoML Model Recommender")
st.markdown("Upload your training and (optional) test dataset, select the target column, and run model selection/tuning.")

# --- File Upload ---
with st.sidebar:
    st.header("1️⃣ Upload Data")
    train_file = st.file_uploader("Upload Training CSV", type="csv", key="train")
    test_file = st.file_uploader("Upload Test CSV (optional)", type="csv", key="test")

# --- Data Loading ---
df_train, df_test = None, None
if train_file is not None:
    df_train = pd.read_csv(train_file)
    st.subheader("Training Data Preview")
    st.dataframe(df_train.head())

if test_file is not None:
    df_test = pd.read_csv(test_file)
    st.subheader("Test Data Preview")
    st.dataframe(df_test.head())

# --- Target Selection & Options ---
if df_train is not None:
    with st.sidebar:
        st.header("2️⃣ Select Target & Options")
        target_col = st.selectbox("Select target column", df_train.columns)
        use_smote = st.checkbox("Use SMOTE (for imbalanced data)", value=True)
        use_polynomial = st.checkbox("Polynomial features", value=False)
        use_interactions = st.checkbox("Interaction features", value=False)
        scoring = st.selectbox("Scoring metric", ["f1", "roc_auc", "accuracy", "precision", "recall"], index=0)
        n_trials = st.number_input("Optuna trials per model", min_value=5, max_value=100, value=10, step=1)
        cv = st.number_input("Cross-validation folds", min_value=3, max_value=10, value=5, step=1)
        model_list = st.multiselect(
            "Models to tune",
            list(TUNING_CONFIGS.keys()),
            default=["RandomForest", "XGBoost", "LightGBM"]
        )
        run_button = st.button("🚀 Run AutoML")

    # --- Run AutoML Pipeline ---
    if run_button:
        st.info("Preprocessing data...")
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        preprocessor, X_train_proc = build_preprocessing_pipeline(
            X_train,
            use_polynomial=use_polynomial,
            use_interactions=use_interactions
        )
        st.success(f"Preprocessing complete. Shape: {X_train_proc.shape}")

        X_test_proc, y_test = None, None
        if df_test is not None:
            if target_col in df_test.columns:
                y_test = df_test[target_col]
                X_test = df_test.drop(columns=[target_col])
            else:
                X_test = df_test.copy()
            X_test_proc = preprocessor.transform(X_test)

        # Baseline leaderboard
        st.info("Evaluating baseline models...")
        leaderboard = evaluate_models(X_train_proc, y_train, cv=cv, scoring=scoring)
        st.subheader("Baseline Model Scores")
        st.dataframe(leaderboard)

        # Tuning
        st.info("Tuning models with Optuna...")
        lb_tuned, tuned_models = tune_multiple_models(
            X_train_proc, y_train,
            model_list=model_list,
            n_trials=n_trials,
            cv=cv,
            cv_n_jobs=1,
            sampler="TPESampler",
            scoring=scoring,
            use_smote=use_smote,
            random_state=42,
        )
        st.subheader("Tuned Model Leaderboard")
        st.dataframe(lb_tuned)

        # Best model
        if not lb_tuned.empty and lb_tuned["Tuned Score"].notnull().any():
            best_row = lb_tuned.iloc[0]
            best_name = best_row["Model"]
            best_model = tuned_models[best_name]
            st.success(f"Best model: {best_name}")

            # SHAP
            feature_names = get_feature_names(preprocessor)
            if X_train_proc.shape[1] <= 8000:
                with st.spinner("Generating SHAP summary plot..."):
                    try:
                        X_df = pd.DataFrame(X_train_proc, columns=feature_names)
                        explain_model(best_model, X_df, feature_names, save_path="models/shap_summary.png")
                        st.image("models/shap_summary.png", caption="SHAP Feature Importance")
                    except Exception as e:
                        st.warning(f"SHAP failed: {e}")

            # Test set evaluation
            if X_test_proc is not None:
                st.info("Evaluating on test set...")
                y_pred = best_model.predict(X_test_proc)
                try:
                    y_proba = best_model.predict_proba(X_test_proc)
                    y_proba_bin = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
                except Exception:
                    y_proba_bin = None

                st.write("Confusion Matrix:")
                plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (Test)")
                if y_proba_bin is not None:
                    st.write("ROC Curve:")
                    plot_roc_curve(y_test, y_proba_bin, title="ROC Curve (Test)")
                st.write("Classification Report:")
                st.text(print_classification_report(y_test, y_pred))

                # Download predictions
                preds_df = pd.DataFrame({
                    "index": df_test.index,
                    "prediction": y_pred
                })
                csv = preds_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Download model
            save_model(best_model, filename="models/best_model.pkl")
            with open("models/best_model.pkl", "rb") as f:
                st.download_button(
                    label="Download Best Model (.pkl)",
                    data=f,
                    file_name="best_model.pkl"
                )
        else:
            st.error("No successful model tuning results.")
