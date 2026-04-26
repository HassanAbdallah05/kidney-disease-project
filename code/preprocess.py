import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = "data/kidney_disease.csv"
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.30  # same 70/30 split as the original paper


# -----------------------------------------------------------------------------
# 2. LOAD DATASET
# -----------------------------------------------------------------------------
def load_data(path):
    """Load the raw CKD dataset from CSV."""
    print(f"[1/7] Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"      Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------------------
# 3. CLEAN DATA
# -----------------------------------------------------------------------------
def clean_data(df):
    """
    Cleans column names, drops the ID column, and fixes
    common typos in categorical values that exist in this dataset.
    """
    print("[2/7] Cleaning data...")

    # Drop the 'id' column if it exists (not useful for prediction)
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Fix known typos in the UCI CKD dataset
    df.replace({"\t?": np.nan, "?": np.nan, "": np.nan}, inplace=True)
    df.replace({"\tno": "no", "\tyes": "yes", " yes": "yes"}, inplace=True)
    df.replace({"ckd\t": "ckd"}, inplace=True)

    # Convert numeric columns that were read as strings back to numbers
    numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu",
                    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------------------------------------------------------
# 4. HANDLE MISSING VALUES (RANDOM IMPUTATION — same as original paper)
# -----------------------------------------------------------------------------
def random_imputation(df):
    """
    Fills missing values by randomly sampling from the existing
    observed values in the same column. This is the same imputation
    method used in the original Singamsetty et al. (2024) paper.
    """
    print("[3/7] Imputing missing values (random sampling)...")
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            observed_values = df[col].dropna().values
            n_missing = df[col].isnull().sum()
            random_values = np.random.choice(observed_values, size=n_missing)
            df.loc[df[col].isnull(), col] = random_values
    return df


# -----------------------------------------------------------------------------
# 5. ENCODE CATEGORICAL FEATURES
# -----------------------------------------------------------------------------
def encode_categorical(df):
    """Convert categorical (text) columns into numeric using LabelEncoder."""
    print("[4/7] Encoding categorical features...")
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# -----------------------------------------------------------------------------
# 6. SPLIT, SCALE, AND APPLY SMOTE
# -----------------------------------------------------------------------------
def split_scale_smote(df, target_col="classification"):
    """
    Splits into train/test, scales numerical features using StandardScaler,
    and applies SMOTE on the training set only (to avoid data leakage).
    """
    print("[5/7] Splitting into train/test...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Scale features (StandardScaler — important for SVM and ANN)
    print("[6/7] Scaling features (StandardScaler)...")
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

    # Apply SMOTE only on the training set
    print("[7/7] Applying SMOTE for class balance...")
    print(f"      Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"      After  SMOTE: {dict(pd.Series(y_train_balanced).value_counts())}")

    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler


# -----------------------------------------------------------------------------
# 7. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  CKD PREPROCESSING PIPELINE")
    print("=" * 60)

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step-by-step pipeline
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = random_imputation(df)
    df, encoders = encode_categorical(df)
    X_train, X_test, y_train, y_test, scaler = split_scale_smote(df)

    # Save the processed datasets
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    pd.Series(y_train).to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    pd.Series(y_test).to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    # Save the scaler and encoders for reuse later (in inference / explainability)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{OUTPUT_DIR}/encoders.pkl")

    print("\n Preprocessing complete!")
    print(f"   Files saved to: {OUTPUT_DIR}/")
    print("   Next step: run feature_selection.py")


if __name__ == "__main__":
    main()