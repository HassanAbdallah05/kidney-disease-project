import os
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Configuration
DATA_PATH = "data/kidney_disease.csv"
RESULTS_DIR = "results"

RANDOM_STATE = 40
N_FOLDS = 10
K_FEATURES = 10


# Compatibility for different sklearn versions
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# Load dataset
def load_data():
    return pd.read_csv(DATA_PATH)


# Clean raw data only
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


# Split features and target
def prepare_xy(df):
    X = df.drop("classification", axis=1)
    y = df["classification"]

    return X, y


# Build preprocessing inside CV pipeline
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        sparse_threshold=0
    )

    return preprocessor


# Define models
def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "ann": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "svm": SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }


# Build full or selected pipeline
def build_pipeline(preprocessor, model, version):
    steps = [
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE))
    ]

    if version == "selected":
        steps.append(
            ("feature_selection", SelectKBest(score_func=f_classif, k=K_FEATURES))
        )

    steps.append(("model", model))

    return Pipeline(steps=steps)


# Evaluate one model/version with CV
def evaluate_model_cv(model_name, model, version, X, y, cv):
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor, model, version)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "mcc": make_scorer(matthews_corrcoef)
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    return {
        "Model": model_name,
        "Version": version,

        "Accuracy_mean": round(scores["test_accuracy"].mean(), 4),
        "Accuracy_std": round(scores["test_accuracy"].std(), 4),

        "Precision_mean": round(scores["test_precision"].mean(), 4),
        "Precision_std": round(scores["test_precision"].std(), 4),

        "Recall_mean": round(scores["test_recall"].mean(), 4),
        "Recall_std": round(scores["test_recall"].std(), 4),

        "F1_mean": round(scores["test_f1"].mean(), 4),
        "F1_std": round(scores["test_f1"].std(), 4),

        "ROC_AUC_mean": round(scores["test_roc_auc"].mean(), 4),
        "ROC_AUC_std": round(scores["test_roc_auc"].std(), 4),

        "MCC_mean": round(scores["test_mcc"].mean(), 4),
        "MCC_std": round(scores["test_mcc"].std(), 4),
    }


# Add paper-friendly columns
def add_paper_columns(results_df):
    results_df["Accuracy"] = results_df.apply(
        lambda row: f"{row['Accuracy_mean']:.4f} ± {row['Accuracy_std']:.4f}",
        axis=1
    )

    results_df["ROC_AUC"] = results_df.apply(
        lambda row: f"{row['ROC_AUC_mean']:.4f} ± {row['ROC_AUC_std']:.4f}",
        axis=1
    )

    results_df["MCC"] = results_df.apply(
        lambda row: f"{row['MCC_mean']:.4f} ± {row['MCC_std']:.4f}",
        axis=1
    )

    return results_df


# Save results
def save_results(results_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df.to_csv(f"{RESULTS_DIR}/cv_results.csv", index=False)

    paper_df = results_df[
        ["Model", "Version", "Accuracy", "ROC_AUC", "MCC"]
    ].copy()

    paper_df.to_csv(f"{RESULTS_DIR}/cv_results_for_paper.csv", index=False)


# Main pipeline
def main():
    print("Loading and preparing data...")

    df = load_data()
    df = clean_data(df)
    df = encode_target(df)

    X, y = prepare_xy(df)

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_models()
    results = []

    for version in ["full", "selected"]:
        print(f"\nRunning leakage-safe {N_FOLDS}-fold CV using {version} features...")

        for model_name, model in models.items():
            print(f"  Cross-validating {model_name}_{version}...", end=" ", flush=True)

            result = evaluate_model_cv(
                model_name=model_name,
                model=model,
                version=version,
                X=X,
                y=y,
                cv=cv
            )

            results.append(result)

            print(
                f"done "
                f"(Acc={result['Accuracy_mean']:.4f} ± {result['Accuracy_std']:.4f}, "
                f"MCC={result['MCC_mean']:.4f} ± {result['MCC_std']:.4f})"
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("No cross-validation results were generated.")

    results_df = results_df.sort_values(
        "MCC_mean",
        ascending=False
    ).reset_index(drop=True)

    results_df = add_paper_columns(results_df)

    save_results(results_df)

    print(f"\n=== Leakage-Safe {N_FOLDS}-Fold Cross-Validation Results Sorted by MCC ===")
    print(
        results_df[
            ["Model", "Version", "Accuracy", "ROC_AUC", "MCC"]
        ].to_string(index=False)
    )

    print("\nCross-validation complete.")
    print(f"Full results saved to: {RESULTS_DIR}/cv_results.csv")
    print(f"Paper table saved to: {RESULTS_DIR}/cv_results_for_paper.csv")


if __name__ == "__main__":
    main()