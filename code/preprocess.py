import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


# Configuration
DATA_PATH = "data/kidney_disease.csv"
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.30


# Load dataset
def load_data(path):
    return pd.read_csv(path)


# Clean dataset
def clean_data(df):
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df.replace({"\t?": np.nan, "?": np.nan, "": np.nan, "nan": np.nan}, inplace=True)
    df.replace({"\tno": "no", "\tyes": "yes", " yes": "yes", "ckd\t": "ckd"}, inplace=True)

    numeric_cols = [
        "age", "bp", "sg", "al", "su", "bgr", "bu",
        "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Impute missing values
def random_imputation(df):
    df = df.copy()
    np.random.seed(RANDOM_STATE)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            observed = df[col].dropna().values
            n_missing = df[col].isnull().sum()
            df.loc[df[col].isnull(), col] = np.random.choice(observed, size=n_missing)

    return df


# Encode categorical features
def encode_categorical(df):
    df = df.copy()
    encoders = {}

    df["classification"] = df["classification"].map({
        "notckd": 0,
        "ckd": 1
    })

    for col in df.select_dtypes(include=["object"]).columns:
        if col != "classification":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


# Split data
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


# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


# Apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


# Save outputs
def save_outputs(X_train, X_test, y_train, y_test, scaler, encoders):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)

    pd.Series(y_train, name="classification").to_csv(
        f"{OUTPUT_DIR}/y_train.csv",
        index=False
    )

    pd.Series(y_test, name="classification").to_csv(
        f"{OUTPUT_DIR}/y_test.csv",
        index=False
    )

    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{OUTPUT_DIR}/encoders.pkl")

    metadata = {
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "imputation_method": "random sampling",
        "scaling_method": "StandardScaler",
        "balancing_method": "SMOTE",
        "target_mapping": {
            "notckd": 0,
            "ckd": 1
        }
    }

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# Main pipeline
def main():
    print("Loading and cleaning data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = random_imputation(df)
    df, encoders = encode_categorical(df)

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    X_train, y_train = apply_smote(X_train, y_train)

    save_outputs(X_train, X_test, y_train, y_test, scaler, encoders)

    print("Preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()