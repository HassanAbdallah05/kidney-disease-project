import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Configuration
DATA_PATH = "data/kidney_disease.csv"
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 40
TEST_SIZE = 0.30


# Load dataset
def load_data(path):
    return pd.read_csv(path)


# Clean raw dataset only
def clean_data(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df.replace(
        {
            "\t?": np.nan,
            "?": np.nan,
            "": np.nan,
            "nan": np.nan,
            "\tno": "no",
            "\tyes": "yes",
            " yes": "yes",
            "ckd\t": "ckd"
        },
        inplace=True
    )

    numeric_cols = [
        "age", "bp", "sg", "al", "su", "bgr", "bu",
        "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Encode target manually
def encode_target(df):
    df = df.copy()

    df["classification"] = df["classification"].map({
        "notckd": 0,
        "ckd": 1
    })

    if df["classification"].isnull().sum() > 0:
        raise ValueError("Target mapping failed. Check classification values.")

    return df


# Split raw data before imputation, scaling, encoding, or SMOTE
def split_data(df, target_col="classification"):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# Save raw train/test split
def save_outputs(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(f"{OUTPUT_DIR}/X_train_raw.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test_raw.csv", index=False)

    pd.Series(y_train, name="classification").to_csv(
        f"{OUTPUT_DIR}/y_train.csv",
        index=False
    )

    pd.Series(y_test, name="classification").to_csv(
        f"{OUTPUT_DIR}/y_test.csv",
        index=False
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    metadata = {
        "n_features_raw": X_train.shape[1],
        "feature_names_raw": X_train.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target_mapping": {
            "notckd": 0,
            "ckd": 1
        },
        "note": (
            "This file only cleans and splits raw data. "
            "Imputation, encoding, scaling, SMOTE, feature selection, "
            "and model training should happen inside training/CV pipelines "
            "to reduce data leakage."
        )
    }

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# Main pipeline
def main():
    print("Loading and cleaning data...")

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_target(df)

    print("Splitting raw train/test data...")

    X_train, X_test, y_train, y_test = split_data(df)

    save_outputs(X_train, X_test, y_train, y_test)

    print("Preprocessing complete.")
    print("Raw train/test files saved to data/processed/")
    print("Target mapping: notckd = 0, ckd = 1")


if __name__ == "__main__":
    main()