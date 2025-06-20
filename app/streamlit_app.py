import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import TUNING_CONFIGS
from pipeline import run_automl

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
    try:
        df_train = pd.read_csv(train_file)
        st.subheader("Training Data Preview")
        st.dataframe(df_train.head())
    except Exception as e:
        st.error(f"Failed to read training CSV: {e}")

if test_file is not None:
    try:
        df_test = pd.read_csv(test_file)
        st.subheader("Test Data Preview")
        st.dataframe(df_test.head())
    except Exception as e:
        st.error(f"Failed to read test CSV: {e}")

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
        with st.spinner("Running AutoML pipeline..."):
            try:
                results = run_automl(
                    df_train=df_train,
                    df_test=df_test,
                    target_col=target_col,
                    use_smote=use_smote,
                    use_polynomial=use_polynomial,
                    use_interactions=use_interactions,
                    scoring=scoring,
                    n_trials=int(n_trials),
                    cv=int(cv),
                    model_list=model_list,
                )
            except Exception as e:
                st.error(f"AutoML pipeline failed: {e}")
                st.stop()

        # Show baseline leaderboard
        if results.get("leaderboard") is not None:
            st.subheader("Baseline Model Scores")
            st.dataframe(results["leaderboard"])

        # Show tuned leaderboard
        if results.get("lb_tuned") is not None:
            st.subheader("Tuned Model Leaderboard")
            st.dataframe(results["lb_tuned"])

        # Best model info
        if results.get("best_name"):
            st.success(f"Best model: {results['best_name']}")

        # SHAP plot
        if results.get("shap_path"):
            st.image(results["shap_path"], caption="SHAP Feature Importance")

        # Test set evaluation
        if results.get("test_predictions_df") is not None:
            st.subheader("Test Set Predictions")
            st.dataframe(results["test_predictions_df"].head())
            # Download predictions
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_pred:
                results["test_predictions_df"].to_csv(tmp_pred.name, index=False)
                tmp_pred.flush()
                with open(tmp_pred.name, "rb") as f:
                    st.download_button(
                        label="Download Predictions CSV",
                        data=f,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

        # Classification report
        if results.get("classification_report"):
            st.subheader("Classification Report")
            st.text(results["classification_report"])

        # Download model
        if results.get("model_path"):
            with open(results["model_path"], "rb") as f:
                st.download_button(
                    label="Download Best Model (.pkl)",
                    data=f,
                    file_name="best_model.pkl"
                )
        # Error handling for model tuning
        if results.get("error"):
            st.error(f"AutoML error: {results['error']}")
