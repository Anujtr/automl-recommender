import optuna
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from config import TUNING_CONFIGS
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

# Add imports for new samplers and SMOTE
try:
    from optuna.integration import BoTorchSampler
    _HAS_BOTORCH = True
except ImportError:
    _HAS_BOTORCH = False

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# ----------------------------------------
# Setup logging to file only
# ----------------------------------------
optuna_log = logging.getLogger("optuna")
optuna_log.setLevel(logging.WARNING)
file_handler = logging.FileHandler("optuna.log")
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
        return make_scorer(f1_score, average=average)

def get_sampler(sampler_name):
    if sampler_name == "TPESampler":
        return TPESampler(n_startup_trials=10, multivariate=True)
    elif sampler_name == "BoTorchSampler":
        if _HAS_BOTORCH:
            return BoTorchSampler()
        else:
            print("⚠️ BoTorchSampler not available, falling back to TPESampler.")
            return TPESampler(n_startup_trials=10, multivariate=True)
    else:
        return TPESampler(n_startup_trials=10, multivariate=True)

def tune_model(
    X, y, model_name="RandomForest", n_trials=30, cv=5, cv_n_jobs=1,
    sampler="TPESampler", scoring="f1", use_smote=False, random_state=42
):
    if model_name not in TUNING_CONFIGS:
        raise ValueError(f"❌ Unsupported model: {model_name}")

    config = TUNING_CONFIGS[model_name]
    Estimator = config["estimator"]
    param_sampler = config["params"]

    scorer = get_scorer(y, scoring)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial):
        params = param_sampler(trial)
        model = Estimator(**params)
        X_res, y_res = X, y
        if use_smote:
            try:
                sm = SMOTE(random_state=random_state)
                X_res, y_res = sm.fit_resample(X, y)
            except Exception:
                pass
        score = cross_val_score(model, X_res, y_res, cv=cv_splitter, scoring=scorer, n_jobs=cv_n_jobs)
        return np.mean(score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction="maximize", sampler=get_sampler(sampler))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[])
        final_model = Estimator(**study.best_params)
        if use_smote:
            sm = SMOTE(random_state=random_state)
            X_fit, y_fit = sm.fit_resample(X, y)
        else:
            X_fit, y_fit = X, y
        final_model.fit(X_fit, y_fit)
        return final_model, study, None
    except Exception as e:
        return None, None, str(e)

def tune_multiple_models(
    X, y, model_list=None, n_trials=30, cv=5, n_jobs=1, cv_n_jobs=1,
    sampler="TPESampler", scoring="f1", use_smote=False, random_state=42,
    export_leaderboard_path=None, export_params_path=None
):
    if model_list is None:
        model_list = list(TUNING_CONFIGS.keys())

    results = []
    tuned_models = {}
    tuned_params = {}

    print(f"\n🛠️  Tuning {len(model_list)} models with {n_trials} trials each...\n")

    def tune_single(name):
        print(f"🔧 Starting tuning for model: {name}")
        model, study, err = tune_model(
            X, y, name, n_trials=n_trials, cv=cv, cv_n_jobs=cv_n_jobs,
            sampler=sampler, scoring=scoring, use_smote=use_smote, random_state=random_state
        )
        if err:
            print(f"❌ Tuning failed for {name}: {err}")
            return {"Model": name, "Tuned Score": np.nan, "Error": err}, name, None, None
        print(f"✅ Finished tuning for {name}. Best Score: {study.best_value:.4f}")
        tuned_params = study.best_params if study else {}
        return {"Model": name, "Tuned Score": study.best_value, "Error": None}, name, model, tuned_params

    # Progress bar for model tuning
    out = []
    with tqdm(total=len(model_list), desc="Tuning models", ncols=80) as pbar:
        for res in Parallel(n_jobs=n_jobs, prefer="threads" if n_jobs != 1 else "processes")(
            delayed(tune_single)(name) for name in model_list
        ):
            out.append(res)
            pbar.update(1)

    for res, name, model, params in out:
        results.append(res)
        if model is not None:
            tuned_models[name] = model
            tuned_params[name] = params

    leaderboard = pd.DataFrame(results).sort_values("Tuned Score", ascending=False).reset_index(drop=True)

    if export_leaderboard_path:
        leaderboard.to_csv(export_leaderboard_path, index=False)
    if export_params_path:
        import json
        with open(export_params_path, "w") as f:
            json.dump(tuned_params, f, indent=2)

    return leaderboard, tuned_models
