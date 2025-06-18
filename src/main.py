# src/main.py

import sys
import pandas as pd
from preprocessing import build_preprocessing_pipeline

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/main.py <path_to_csv> <target_column>")
        sys.exit(1)

    csv_path = sys.argv[1]
    target_col = sys.argv[2]

    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in dataset.")
        sys.exit(1)

    print(f"✅ Dataset loaded with shape: {df.shape}")
    print(f"🎯 Target column: {target_col}\n")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    pipeline, X_processed = build_preprocessing_pipeline(X)

    print(f"✅ Preprocessing complete. Processed shape: {X_processed.shape}")
    print("🎉 Ready for model selection and tuning!")

if __name__ == "__main__":
    main()
