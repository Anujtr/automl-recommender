import pandas as pd
from pathlib import Path

def get_feature_names(preprocessor):
    """
    Return list of feature names saved during preprocessing.
    Safe fallback to generic f0..fn if missing.
    """
    return getattr(preprocessor, "feature_names_", [f"f{i}" for i in range(getattr(preprocessor, 'n_features_in_', 0))])

def load_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")
    return pd.read_csv(path_obj)