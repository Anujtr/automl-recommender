# src/model_selector.py

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score
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
    scoring_func = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average="binary" if len(set(y)) == 2 else "macro")
    }.get(scoring, make_scorer(f1_score))

    results = []

    for name, model in MODEL_CANDIDATES.items():
        clean_name = name.replace("LogisticRegression", "Logistic Regression") \
                         .replace("RandomForest", "Random Forest") \
                         .replace("SVM", "Support Vector Machine") \
                         .replace("KNN", "K-Nearest Neighbors") \
                         .replace("MLP", "Neural Net (MLP)")
        print(f"🔍 Evaluating {clean_name}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring_func)
        results.append({
            "Model": clean_name,
            "Mean Score": scores.mean(),
            "Std Dev": scores.std()
        })

    leaderboard = pd.DataFrame(results).sort_values(by="Mean Score", ascending=False)
    return leaderboard.reset_index(drop=True)
