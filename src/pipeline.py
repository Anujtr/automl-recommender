import pandas as pd
import numpy as np
import tempfile
import os
import traceback

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

def run_automl(
    df_train,
    df_test,
    target_col,
    use_smote,
    use_polynomial,
    use_interactions,
    scoring,
    n_trials,
    cv,
    model_list,
):
    result = {
        "leaderboard": None,
        "lb_tuned": None,
        "tuned_models": None,
        "best_model": None,
        "best_name": None,
        "shap_path": None,
        "test_predictions_df": None,
        "classification_report": None,
        "model_path": None,
        "error": None,
    }
    try:
        # --- Validation ---
        if target_col not in df_train.columns:
            result["error"] = f"Target column '{target_col}' not found in training data."
            return result
        if df_train[target_col].isnull().any():
            result["error"] = "Missing values in target column. Please clean your data."
            return result

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        # Check for non-numeric columns that can't be processed
        if X_train.select_dtypes(include=["object"]).shape[1] > 0:
            # Will be handled by preprocessing, but warn if all columns are object
            if X_train.select_dtypes(exclude=["object"]).shape[1] == 0:
                result["error"] = "All features are non-numeric. Please provide at least one numeric feature."
                return result

        # --- Preprocessing ---
        try:
            preprocessor, X_train_proc = build_preprocessing_pipeline(
                X_train,
                use_polynomial=use_polynomial,
                use_interactions=use_interactions
            )
        except Exception as e:
            result["error"] = f"Preprocessing failed: {e}"
            return result

        # --- Test set preprocessing ---
        X_test_proc, y_test = None, None
        if df_test is not None:
            # Structure check
            if not set(X_train.columns).issubset(set(df_test.columns)):
                result["error"] = "Test set columns do not match training set features."
                return result
            if target_col in df_test.columns:
                y_test = df_test[target_col]
                X_test = df_test.drop(columns=[target_col])
            else:
                X_test = df_test.copy()
            try:
                X_test_proc = preprocessor.transform(X_test)
            except Exception as e:
                result["error"] = f"Test set preprocessing failed: {e}"
                return result

        # --- Baseline leaderboard ---
        try:
            leaderboard = evaluate_models(X_train_proc, y_train, cv=cv, scoring=scoring)
            result["leaderboard"] = leaderboard
        except Exception as e:
            result["error"] = f"Baseline model evaluation failed: {e}"
            return result

        # --- Tuning ---
        try:
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
            result["lb_tuned"] = lb_tuned
            result["tuned_models"] = tuned_models
        except Exception as e:
            result["error"] = f"Model tuning failed: {e}"
            return result

        # --- Best model selection ---
        if lb_tuned is None or lb_tuned.empty or lb_tuned["Tuned Score"].isnull().all():
            result["error"] = "No successful model tuning results."
            return result
        best_row = lb_tuned.iloc[0]
        best_name = best_row["Model"]
        if best_name not in tuned_models:
            result["error"] = f"Best model '{best_name}' not found in tuned models."
            return result
        best_model = tuned_models[best_name]
        result["best_model"] = best_model
        result["best_name"] = best_name

        # --- SHAP explainability ---
        shap_path = None
        feature_names = get_feature_names(preprocessor)
        if X_train_proc.shape[1] <= 8000:
            try:
                X_df = pd.DataFrame(X_train_proc, columns=feature_names)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_shap:
                    explain_model(best_model, X_df, feature_names, save_path=tmp_shap.name)
                    shap_path = tmp_shap.name
            except Exception:
                shap_path = None
        result["shap_path"] = shap_path

        # --- Test set evaluation ---
        test_predictions_df = None
        classification_report_str = None
        if X_test_proc is not None and y_test is not None:
            try:
                y_pred = best_model.predict(X_test_proc)
                try:
                    y_proba = best_model.predict_proba(X_test_proc)
                    y_proba_bin = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
                except Exception:
                    y_proba_bin = None

                # Confusion matrix and ROC curve are not shown here (handled in main app)
                from sklearn.metrics import classification_report
                classification_report_str = classification_report(y_test, y_pred)
                test_predictions_df = pd.DataFrame({
                    "index": df_test.index,
                    "prediction": y_pred
                })
            except Exception as e:
                classification_report_str = f"Test set evaluation failed: {e}"
        elif X_test_proc is not None:
            # Unlabeled test set
            try:
                y_pred = best_model.predict(X_test_proc)
                test_predictions_df = pd.DataFrame({
                    "index": df_test.index,
                    "prediction": y_pred
                })
            except Exception as e:
                test_predictions_df = None

        result["test_predictions_df"] = test_predictions_df
        result["classification_report"] = classification_report_str

        # --- Save model to temp file for download ---
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
                import joblib
                joblib.dump(best_model, tmp_model.name)
                result["model_path"] = tmp_model.name
        except Exception as e:
            result["model_path"] = None

        return result

    except Exception as e:
        result["error"] = f"Unexpected error: {e}\n{traceback.format_exc()}"
        return result
