import optuna
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from config import TUNING_CONFIGS
from joblib import Parallel, delayed


# ----------------------------------------
# Setup logging to file only
# ----------------------------------------
optuna_log = logging.getLogger("optuna")
optuna_log.setLevel(logging.WARNING)

file_handler = logging.FileHandler("optuna.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
optuna_log.addHandler(file_handler)

# ----------------------------------------
# Scoring function
# ----------------------------------------
def get_scorer(y):
    average = "binary" if len(set(y)) == 2 else "macro"
    return make_scorer(f1_score, average=average)

def tune_model(X, y, model_name="RandomForest", n_trials=30, cv=5):
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

    # Logging to file only
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study = optuna.create_study(direction="maximize")
        # Suppress Optuna's progress bar and output
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[])
        final_model = Estimator(**study.best_params)
        final_model.fit(X, y)
        return final_model, study, None
    except Exception as e:
        return None, None, str(e)

def tune_multiple_models(X, y, model_list=None, n_trials=30, cv=5, n_jobs=1):
    if model_list is None:
        model_list = list(TUNING_CONFIGS.keys())

    results = []
    tuned_models = {}

    print(f"\n🛠️  Tuning {len(model_list)} models with {n_trials} trials each...\n")

    def tune_single(name):
        # No print statements here
        model, study, err = tune_model(X, y, name, n_trials=n_trials, cv=cv)
        if err:
            return {"Model": name, "Tuned F1": np.nan, "Error": err}, name, None
        return {"Model": name, "Tuned F1": study.best_value, "Error": None}, name, model

    # Only show a single progress bar for the outer loop
    out = []
    for res in tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads" if n_jobs != 1 else "processes")(
            delayed(tune_single)(name) for name in model_list
        ),
        total=len(model_list),
        desc="Tuning models",
        ncols=80
    ):
        out.append(res)

    for res, name, model in out:
        results.append(res)
        if model is not None:
            tuned_models[name] = model

    leaderboard = pd.DataFrame(results).sort_values("Tuned F1", ascending=False).reset_index(drop=True)
    return leaderboard, tuned_models
