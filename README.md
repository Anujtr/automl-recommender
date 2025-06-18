# AutoML Recommender System

A modular AutoML system that selects, trains, tunes, and evaluates the best machine learning models for structured datasets. Designed for extensibility, experimentation, and integration into full-stack apps.

---

## Features

- 📊 Automatic model selection from multiple classifiers
- ⚙️ Hyperparameter optimization with cross-validation
- 🔍 Evaluation metrics, visualizations, and leaderboard
- 🧹 Preprocessing pipeline (cleaning, encoding, scaling)
- 🧪 Jupyter notebooks for experimentation
- 🌐 Optional Streamlit interface for interactive use

---

## Project Structure

automl-recommender/
│
├── data/ # Raw datasets (ignored by Git)
├── models/ # Trained model binaries (Pickle)
├── notebooks/ # Jupyter notebooks for prototyping
├── src/ # Core Python modules
│ ├── automl_engine.py
│ ├── model_selector.py
│ ├── preprocessing.py
│ └── ...
├── app/ # Streamlit app (optional)
├── tests/ # Unit tests
├── requirements.txt # pip-based dependencies
├── environment.yml # Conda environment definition
├── Dockerfile # Optional container setup
├── .env # Environment variables (ignored)
└── README.md


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
