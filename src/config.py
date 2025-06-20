from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model candidates for baseline evaluation
MODEL_CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced'),
    "SVM": SVC(probability=True, class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=2000),  # Increased from 1000
    "XGBoost": XGBClassifier(n_estimators=100, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=100, verbose=0, auto_class_weights='Balanced')
}

# Tuning configs for Optuna
TUNING_CONFIGS = {
    "RandomForest": {
        "estimator": RandomForestClassifier,
        "params": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
            "class_weight": "balanced"
        }
    },
    "LogisticRegression": {
        "estimator": LogisticRegression,
        "params": lambda trial: {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            "max_iter": 2000,  # Increased from 1000
            "class_weight": "balanced"
        }
    },
    "SVM": {
    "estimator": lambda **kwargs: SVC(probability=True, class_weight='balanced', **kwargs),
    "params": lambda trial: {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    },
    "MLP": {
        "estimator": MLPClassifier,
        "params": lambda trial: {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 64)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 2000  # Increased from 1000
        }
    },
    "KNN": {
        "estimator": KNeighborsClassifier,
        "params": lambda trial: {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2)
        }
    },
    "XGBoost": {
        # wrap estimator so mandatory kwargs are always present
        "estimator": lambda **kwargs: XGBClassifier(
            eval_metric="logloss",
            n_jobs=-1,
            **kwargs
        ),
        "params": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth":    trial.suggest_int("max_depth", 3, 12),
            "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":    trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        },
    },
    "LightGBM": {
        "estimator": LGBMClassifier,
        "params": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "force_col_wise": True,
            "n_jobs": -1,
            "class_weight": "balanced",
            "verbose": -1
        }
    },
    "CatBoost": {
        "estimator": CatBoostClassifier,
        "params": lambda trial: {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": "Balanced",
            "verbose": 0
        }
    }
}
