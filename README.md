# AutoML Recommender System

A modular AutoML system that selects, tunes, and evaluates the best machine learning models for structured datasets. Designed for extensibility, experimentation, and integration into full-stack apps.

---

## Features

- 📊 Automatic model selection from multiple classifiers
- ⚙️ Hyperparameter optimization with cross-validation (Optuna)
- 🔍 Evaluation metrics, visualizations, and leaderboard
- 🧹 Preprocessing pipeline (cleaning, encoding, scaling, polynomial/interactions)
- 🧪 Jupyter notebooks for experimentation
- 🌐 Optional Streamlit interface for interactive use
- 📝 **Exportable `.txt` report** summarizing baseline/tuned scores, confusion matrix, ROC AUC, best hyperparameters, SHAP insights, and ensemble performance

---

## 🔧 Installation

### 📦 Conda (Recommended)

```bash
# Clone repo
git clone https://github.com/Anujtr/automl-recommender.git
cd automl-recommender

# Create Conda environment
conda env create -f environment.yml
conda activate automl

# Optional: Add Jupyter kernel
python -m ipykernel install --user --name=automl --display-name "Python (automl)"
```

---

## Usage

### CLI

```bash
python src/main.py <train_csv> <target_column> [test_csv] [--config config.yaml] [options]
```

**Key options:**
- `--n_trials`: Number of Optuna trials per model
- `--n_jobs`: Number of parallel jobs for tuning
- `--cv_n_jobs`: Number of parallel jobs for cross-validation
- `--models`: List of models to tune (overrides config)
- `--scoring`: Scoring metric (f1, roc_auc, precision, etc.)
- `--sampler`: Optuna sampler (TPESampler, BoTorchSampler, etc.)
- `--use_smote`: Enable SMOTE oversampling
- `--use_polynomial`: Enable polynomial features
- `--use_interactions`: Enable interaction features
- `--export_leaderboard`: Export leaderboard CSV
- `--export_params`: Export tuned params JSON
- `--ensemble_models`: List of models to use in ensemble (overrides config)
- `--ensemble_voting`: Voting type for ensemble (soft/hard)

### Output

- **report.txt**: Exported summary report with baseline/tuned scores, confusion matrix, ROC AUC, best hyperparameters, SHAP insights, and ensemble performance.
- **models/**: Directory containing the best model and ensemble model pickles.
- **predictions.csv**: Predictions for unlabeled test sets.
- **ensemble_predictions.csv**: Ensemble predictions for unlabeled test sets.
- **models/shap_summary.png**: SHAP summary plot for the best model.

---

## Notes

- CatBoost may create a `catboost_info/` directory with logs. This is ignored by default in `.gitignore`.
- For full feature usage, see `config.yaml` and run with `python src/main.py --help`.
- The system supports both YAML and JSON config files.
- SHAP explanations and summary are included in the report if the number of features is not too large.

---
