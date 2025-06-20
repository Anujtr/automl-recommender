# src/model_selector.py

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import warnings
from config import MODEL_CANDIDATES

warnings.filterwarnings("ignore")


def evaluate_models(X, y, cv=5, scoring="f1"):
    """
    Evaluates each model using cross-validation.

    Args:
        X: Features (already preprocessed)
        y: Target
        cv: Number of folds
        scoring: Metric to optimize ("accuracy", "f1", etc.)

    Returns:
        Pandas DataFrame with model names and scores
    """
    average = "binary" if len(set(y)) == 2 else "macro"
    scoring_func = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average=average),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "precision": make_scorer(precision_score, average=average),
        "recall": make_scorer(recall_score, average=average)
    }.get(scoring, make_scorer(f1_score, average=average))

    results = []
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in MODEL_CANDIDATES.items():
        clean_name = name.replace("LogisticRegression", "Logistic Regression") \
                         .replace("RandomForest", "Random Forest") \
                         .replace("SVM", "Support Vector Machine") \
                         .replace("KNN", "K-Nearest Neighbors") \
                         .replace("MLP", "Neural Net (MLP)") \
                         .replace("XGBoost", "XGBoost") \
                         .replace("LightGBM", "LightGBM") \
                         .replace("CatBoost", "CatBoost")
        print(f"🔍 Evaluating {clean_name}...")
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring_func)
        results.append({
            "Model": clean_name,
            "Mean Score": scores.mean(),
            "Std Dev": scores.std()
        })

    leaderboard = pd.DataFrame(results).sort_values(by="Mean Score", ascending=False)
    return leaderboard.reset_index(drop=True)
