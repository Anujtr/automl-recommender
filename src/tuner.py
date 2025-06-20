import logging
import os
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from config import TUNING_CONFIGS
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

try:
    from optuna.integration import BoTorchSampler
    _HAS_BOTORCH = True
except ImportError:
    _HAS_BOTORCH = False

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
import numpy as np
import pandas as pd

# ----------------------------------------
# Setup logging to file only (logs/optuna.log)
# ----------------------------------------
os.makedirs("logs", exist_ok=True)
optuna_log = logging.getLogger("optuna")
optuna_log.setLevel(logging.WARNING)
file_handler = logging.FileHandler("logs/optuna.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
if not any(isinstance(h, logging.FileHandler) for h in optuna_log.handlers):
    optuna_log.addHandler(file_handler)

# ----------------------------------------
# Scoring function
# ----------------------------------------
def get_scorer(y, scoring):
    average = "binary" if len(set(y)) == 2 else "macro"
    if scoring == "f1":
        return make_scorer(f1_score, average=average)
    elif scoring == "roc_auc":
        return make_scorer(roc_auc_score, needs_proba=True)
    elif scoring == "precision":
        return make_scorer(precision_score, average=average)
    elif scoring == "recall":
        return make_scorer(recall_score, average=average)
    elif scoring == "accuracy":
        return make_scorer(accuracy_score)
    else:
        # Default to f1
        return make_scorer(f1_score, average=average)

def get_sampler(sampler_name):
    if sampler_name == "TPESampler":
        return TPESampler()
    elif sampler_name == "BoTorchSampler":
        if _HAS_BOTORCH:
            return BoTorchSampler()
        else:
            raise ImportError("BoTorchSampler requires optuna-integration[botorch]")
    else:
        return TPESampler()

def tune_model(
    X, y, model_name="RandomForest", n_trials=30, cv=5, cv_n_jobs=1,
    sampler="TPESampler", scoring="f1", use_smote=False, random_state=42
):
    if model_name not in TUNING_CONFIGS:
        raise ValueError(f"Model '{model_name}' not in TUNING_CONFIGS.")

    config = TUNING_CONFIGS[model_name]
    Estimator = config["estimator"]
    param_sampler = config["params"]

    scorer = get_scorer(y, scoring)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial):
        params = param_sampler(trial)
        # Some estimators are callables (e.g., XGBoost, SVM)
        if callable(Estimator):
            model = Estimator(**params)
        else:
            model = Estimator(**params)
        X_train, y_train = X, y
        if use_smote:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X, y)
        try:
            scores = cross_val_score(
                model, X_train, y_train, cv=cv_splitter, scoring=scorer, n_jobs=cv_n_jobs
            )
            return np.mean(scores)
        except Exception as e:
            return float("-inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=get_sampler(sampler))
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        # Some estimators are callables (e.g., XGBoost, SVM)
        if callable(Estimator):
            best_model = Estimator(**{**best_params})
        else:
            best_model = Estimator(**{**best_params})
        if use_smote:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X, y)
        else:
            X_train, y_train = X, y
        best_model.fit(X_train, y_train)
        return {
            "score": study.best_value,
            "model": best_model,
            "params": best_params,
            "error": None
        }
    except Exception as e:
        return {
            "score": None,
            "model": None,
            "params": {},
            "error": str(e)
        }

def tune_multiple_models(
    X, y, model_list=None, n_trials=30, cv=5, n_jobs=1, cv_n_jobs=1,
    sampler="TPESampler", scoring="f1", use_smote=False, random_state=42,
    export_leaderboard_path=None, export_params_path=None
):
    if model_list is None:
        model_list = list(TUNING_CONFIGS.keys())

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    results = []
    tuned_models = {}
    tuned_params = {}

    print(f"\n🛠️  Tuning {len(model_list)} models with {n_trials} trials each...\n")

    def tune_single(name):
        try:
            res = tune_model(
                X, y, model_name=name, n_trials=n_trials, cv=cv, cv_n_jobs=cv_n_jobs,
                sampler=sampler, scoring=scoring, use_smote=use_smote, random_state=random_state
            )
            return (res, name, res["model"], res["params"])
        except Exception as e:
            return ({"score": None, "model": None, "params": {}, "error": str(e)}, name, None, {})

    out = []
    with tqdm(total=len(model_list), desc="Tuning models", ncols=80) as pbar:
        for name in model_list:
            res = tune_single(name)
            out.append(res)
            pbar.update(1)

    for res, name, model, params in out:
        results.append({
            "Model": name,
            "Tuned Score": res["score"],
            "Error": res["error"]
        })
        if model is not None:
            tuned_models[name] = model
            tuned_params[name] = params

    leaderboard = pd.DataFrame(results).sort_values(by="Tuned Score", ascending=False).reset_index(drop=True)

    # Save leaderboard and params if requested
    if export_leaderboard_path:
        leaderboard.to_csv(export_leaderboard_path, index=False)
    if export_params_path:
        import json
        with open(export_params_path, "w") as f:
            json.dump(tuned_params, f, indent=2)

    return leaderboard, tuned_models
