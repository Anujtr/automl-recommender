import sys
import os
import pandas as pd
from pathlib import Path

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

def run_tuning_and_select_best(X_train_proc, y_train, preprocessor, model_list):
    print("\n🤖 Auto-tuning multiple models...\n")
    lb_tuned, tuned_models = tune_multiple_models(
        X_train_proc, y_train,
        model_list=model_list,
        n_trials=30,
        n_jobs=os.cpu_count()-2 if os.cpu_count() > 2 else 1,
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
    # ...existing code...
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
    if len(sys.argv) < 3:
        print(
            "Usage: python src/main.py <train_csv> <target_column> [test_csv]",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path = sys.argv[1]
    target_col = sys.argv[2]
    test_path  = sys.argv[3] if len(sys.argv) >= 4 else None

    X_train_proc, y_train, preprocessor, df_test, X_test_proc, y_test = load_and_preprocess(
        train_path, target_col, test_path
    )
    run_baselines(X_train_proc, y_train)
    model_list = ["RandomForest", "SVM", "MLP", "LogisticRegression", "KNN"]
    best_model, best_name = run_tuning_and_select_best(X_train_proc, y_train, preprocessor, model_list)
    evaluate_model(best_model, best_name, df_test, X_test_proc, y_test, target_col)

if __name__ == "__main__":
    main()
