# src/model_selector.py

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Define models to try
MODEL_CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Net (MLP)": MLPClassifier(max_iter=1000)
}

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
        print(f"🔍 Evaluating {name}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring_func)
        results.append({
            "Model": name,
            "Mean Score": scores.mean(),
            "Std Dev": scores.std()
        })

    leaderboard = pd.DataFrame(results).sort_values(by="Mean Score", ascending=False)
    return leaderboard.reset_index(drop=True)
