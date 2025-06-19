import sys
import os
import pandas as pd
from pathlib import Path
import argparse
import yaml
import json

from preprocessing import build_preprocessing_pipeline
from model_selector import evaluate_models
from tuner import tune_multiple_models
from utils import get_feature_names, load_dataset
from explainer import explain_model
from evaluator import (
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
    save_model,
)

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

def load_and_preprocess(train_path, target_col, test_path=None):
    print(f"\n📂 Loading training data from: {train_path}")
    df_train = load_dataset(train_path)
    if target_col not in df_train.columns:
        sys.exit(f"❌ Target column '{target_col}' not found in training set.")

    df_test = None
    if test_path:
        print(f"📂 Loading test data from: {test_path}")
        df_test = load_dataset(test_path)

    print(f"✅ Train shape: {df_train.shape}")
    if df_test is not None:
        print(f"✅ Test  shape: {df_test.shape}")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    preprocessor, X_train_proc = build_preprocessing_pipeline(X_train)
    print(f"✅ Preprocessing complete. Train processed shape: {X_train_proc.shape}")

    X_test_proc, y_test = None, None
    if df_test is not None:
        if target_col in df_test.columns:
            y_test = df_test[target_col]
            X_test = df_test.drop(columns=[target_col])
        else:
            X_test = df_test.copy()
        X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, y_train, preprocessor, df_test, X_test_proc, y_test

def run_baselines(X_train_proc, y_train):
    print("\n📊 Running model evaluation on training set...\n")
    leaderboard = evaluate_models(X_train_proc, y_train, scoring="f1")
    print(leaderboard)
    return leaderboard

def run_tuning_and_select_best(X_train_proc, y_train, preprocessor, model_list, n_trials=30, n_jobs=1, cv_n_jobs=1):
    print("\n🤖 Auto-tuning multiple models...\n")
    lb_tuned, tuned_models = tune_multiple_models(
        X_train_proc, y_train,
        model_list=model_list,
        n_trials=n_trials,
        n_jobs=n_jobs,
        cv_n_jobs=cv_n_jobs,
    )
    print("\n🏆 Tuned Leaderboard")
    print(lb_tuned[["Model", "Tuned F1"]])
    if lb_tuned["Error"].notnull().any():
        print("\n⚠️  Some models failed during tuning:")
        print(lb_tuned[lb_tuned["Error"].notnull()][["Model", "Error"]])

    best_row = lb_tuned.iloc[0]
    best_name = best_row["Model"]
    if best_name not in tuned_models:
        print(f"\n❌ Best model '{best_name}' failed during tuning. Exiting.")
        sys.exit(1)
    best_model = tuned_models[best_name]
    print(f"\n🥇 Best model after tuning: {best_name}")

    # SHAP explainability
    feature_names = get_feature_names(preprocessor)
    if X_train_proc.shape[1] <= 8000:
        try:
            X_df = pd.DataFrame(X_train_proc, columns=feature_names)
            explain_model(best_model, X_df, feature_names)
        except Exception as e:
            print(f"⚠️  SHAP failed: {e}")
    else:
        print("⚠️  Too many features for SHAP – skipping.")

    return best_model, best_name

def evaluate_model(best_model, best_name, df_test, X_test_proc, y_test, target_col):
    if y_test is not None:
        print("\n🧪 Evaluating on hold-out test set...\n")
        y_pred = best_model.predict(X_test_proc)
        y_proba = best_model.predict_proba(X_test_proc)
        y_proba_bin = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (Hold-out)")
        plot_roc_curve(y_test, y_proba_bin, title="ROC Curve (Hold-out)")
        print_classification_report(y_test, y_pred)
    elif X_test_proc is not None:
        print("\n📤 Generating predictions for unlabeled test set...")
        preds = best_model.predict(X_test_proc)
        out = pd.DataFrame({
            "PassengerId": df_test.get("PassengerId", pd.RangeIndex(len(preds))),
            target_col: preds,
        })
        out.to_csv("predictions.csv", index=False, columns=["PassengerId", target_col])
        print("✅ Saved predictions to predictions.csv")
    save_model(best_model, filename="models/best_model.pkl")

def main():
    parser = argparse.ArgumentParser(description="AutoML Recommender")
    parser.add_argument("train_csv", nargs="?", help="Path to training CSV")
    parser.add_argument("target_col", nargs="?", help="Target column name")
    parser.add_argument("test_csv", nargs="?", default=None, help="Path to test CSV (optional)")
    parser.add_argument("--config", default="config.yaml", help="YAML/JSON config file (default: config.yaml)")
    parser.add_argument("--n_trials", type=int, help="Number of Optuna trials per model")
    parser.add_argument("--n_jobs", type=int, help="Number of parallel jobs for tuning")
    parser.add_argument("--cv_n_jobs", type=int, help="Number of parallel jobs for cross-validation")
    parser.add_argument("--models", nargs="+", help="List of models to tune (overrides config)")
    args = parser.parse_args()

    # Load config file if present
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)

    # CLI overrides config
    train_path = args.train_csv or config.get("train_csv")
    target_col = args.target_col or config.get("target_col")
    test_path = args.test_csv or config.get("test_csv")
    n_trials = args.n_trials or config.get("n_trials", 30)
    n_jobs = args.n_jobs or config.get("n_jobs", (os.cpu_count()-2 if os.cpu_count() > 2 else 1))
    cv_n_jobs = args.cv_n_jobs or config.get("cv_n_jobs", 1)
    model_list = args.models or config.get("models", ["RandomForest", "SVM", "MLP", "LogisticRegression", "KNN", "XGBoost"])

    if not train_path or not target_col:
        print(
            "Usage: python src/main.py <train_csv> <target_column> [test_csv] [--config config.yaml]",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train_proc, y_train, preprocessor, df_test, X_test_proc, y_test = load_and_preprocess(
        train_path, target_col, test_path
    )
    run_baselines(X_train_proc, y_train)
    best_model, best_name = run_tuning_and_select_best(
        X_train_proc, y_train, preprocessor, model_list, n_trials=n_trials, n_jobs=n_jobs, cv_n_jobs=cv_n_jobs
    )
    evaluate_model(best_model, best_name, df_test, X_test_proc, y_test, target_col)

if __name__ == "__main__":
    main()
