# src/tuner.py

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import numpy as np

def tune_random_forest(X, y, n_trials=50, cv=5, scoring="f1"):
    """
    Tunes a RandomForestClassifier using Optuna.

    Args:
        X: Preprocessed feature matrix
        y: Target vector
        n_trials: Number of optimization trials
        cv: Number of CV folds
        scoring: Metric to optimize

    Returns:
        Best model and Optuna study object
    """

    # Define scoring function
    scorer = make_scorer(f1_score, average="binary" if len(set(y)) == 2 else "macro")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }

        clf = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(clf, X, y, cv=cv, scoring=scorer)
        return np.mean(score)

    print(f"🔧 Starting Random Forest hyperparameter tuning with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X, y)

    print("✅ Tuning complete. Best params:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    return best_model, study
