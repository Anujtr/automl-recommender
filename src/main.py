import sys
import pandas as pd
from pathlib import Path

from preprocessing import build_preprocessing_pipeline
from model_selector import evaluate_models
from tuner import tune_random_forest
from evaluator import (
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
    save_model,
)

def load_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        sys.exit(f"❌ File not found: {path}")
    return pd.read_csv(path_obj)

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

    # ------------------------------------------------------------------ #
    # 1️⃣ LOAD DATASETS
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2️⃣ PREPROCESSING (fit on train, transform train & test)
    # ------------------------------------------------------------------ #
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
            X_test = df_test.copy()  # unlabeled competition test set
        X_test_proc = preprocessor.transform(X_test)

    # ------------------------------------------------------------------ #
    # 3️⃣ BASELINE MODEL EVALUATION
    # ------------------------------------------------------------------ #
    print("\n📊 Running model evaluation on training set...\n")
    leaderboard = evaluate_models(X_train_proc, y_train, scoring="f1")
    print(leaderboard)

    # ------------------------------------------------------------------ #
    # 4️⃣ HYPERPARAMETER TUNING
    # ------------------------------------------------------------------ #
    print("\n🎯 Tuning Random Forest with Optuna...\n")
    best_model, _ = tune_random_forest(X_train_proc, y_train, n_trials=30)

    # ------------------------------------------------------------------ #
    # 5️⃣ FINAL EVALUATION OR PREDICTION
    # ------------------------------------------------------------------ #
    if y_test is not None:
        print("\n🧪 Evaluating on hold-out test set...\n")
        y_pred  = best_model.predict(X_test_proc)
        y_proba = best_model.predict_proba(X_test_proc)[:, 1]

        plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (Hold-out)")
        plot_roc_curve(y_test, y_proba, title="ROC Curve (Hold-out)")
        print_classification_report(y_test, y_pred)
    elif X_test_proc is not None:
        print("\n📤 Generating predictions for unlabeled test set...")
        preds = best_model.predict(X_test_proc)
        out = pd.DataFrame({
            "PassengerId": df_test.get("PassengerId", pd.RangeIndex(len(preds))),
            target_col: preds,
        })
        out.to_csv("predictions.csv", index=False)
        print("✅ Saved predictions to predictions.csv")

    # ------------------------------------------------------------------ #
    # 6️⃣ SAVE BEST MODEL
    # ------------------------------------------------------------------ #
    save_model(best_model, filename="models/best_model.pkl")


if __name__ == "__main__":
    main()
