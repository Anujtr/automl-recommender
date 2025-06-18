import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# ----------------------------------------
# Scoring function
# ----------------------------------------
def get_scorer(y):
    average = "binary" if len(set(y)) == 2 else "macro"
    return make_scorer(f1_score, average=average)

# ----------------------------------------
# Search space configs
# ----------------------------------------
TUNING_CONFIGS = {
    "RandomForest": {
        "estimator": RandomForestClassifier,
        "params": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }
    },
    "LogisticRegression": {
        "estimator": LogisticRegression,
        "params": lambda trial: {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            "max_iter": 1000
        }
    },
    "SVM": {
        "estimator": SVC,
        "params": lambda trial: {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True
        }
    },
    "MLP": {
        "estimator": MLPClassifier,
        "params": lambda trial: {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 64)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 1000
        }
    }
}

def tune_model(X, y, model_name="RandomForest", n_trials=30, cv=5, n_jobs=1):
    if model_name not in TUNING_CONFIGS:
        raise ValueError(f"❌ Unsupported model: {model_name}")

    config = TUNING_CONFIGS[model_name]
    Estimator = config["estimator"]
    param_sampler = config["params"]

    scorer = get_scorer(y)

    def objective(trial):
        params = param_sampler(trial)
        model = Estimator(**params)
        score = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        return np.mean(score)

    print(f"🔧 Tuning {model_name} with Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    print(f"🏁 Best score for {model_name}: {study.best_value:.4f}")

    print(f"✅ Best trial for {model_name}:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    final_model = Estimator(**study.best_params)
    final_model.fit(X, y)
    return final_model, study

def tune_multiple_models(X, y, model_list=None, n_trials=30, cv=5, n_jobs=1):
    if model_list is None:
        model_list = list(TUNING_CONFIGS.keys())

    results = []
    tuned_models = {}

    print(f"\n🛠️  Tuning {len(model_list)} models with {n_trials} trials each (n_jobs={n_jobs})...\n")

    for name in tqdm(model_list, desc="Tuning models"):
        try:
            model, study = tune_model(X, y, name, n_trials=n_trials, cv=cv, n_jobs=n_jobs)
            tuned_models[name] = model
            results.append({"Model": name, "Tuned F1": study.best_value})
        except Exception as e:
            print(f"⚠️  Skipping {name} due to error: {e}")

    leaderboard = pd.DataFrame(results).sort_values("Tuned F1", ascending=False).reset_index(drop=True)
    return leaderboard, tuned_models
