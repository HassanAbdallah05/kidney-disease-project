import os
import json
import time
import warnings
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Configuration
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved_models"
RESULTS_DIR = "results"

RANDOM_STATE = 40
CV_FOLDS = 5
K_FEATURES = 10


# Compatibility for different sklearn versions
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# Load raw training data
def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_raw.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").iloc[:, 0]

    return X_train, y_train


# Build preprocessing inside pipeline
def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

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

    return preprocessor, numeric_features, categorical_features


# Define models and hyperparameter grids
def get_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"]
            }
        ),

        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5]
            }
        ),

        "extra_trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20]
            }
        ),

        "lightgbm": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [-1, 10]
            }
        ),

        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10]
            }
        ),

        "ann": (
            MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.01]
            }
        ),

        "svm": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        ),

        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 6, 10]
            }
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


# Train one model
def train_model(pipeline, param_grid, X_train, y_train):
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=make_scorer(matthews_corrcoef),
        n_jobs=-1
    )

    start_time = time.time()
    grid.fit(X_train, y_train)
    training_time = time.time() - start_time

    return grid.best_estimator_, grid.best_params_, grid.best_score_, training_time


# Train all models for one version
def train_all_models(version, X_train, y_train, preprocessor):
    print(f"\nTraining models using {version} features")

    models = get_models()
    summary = {}

    for name, (model, param_grid) in models.items():
        print(f"  Training {name}_{version}...", end=" ", flush=True)

        pipeline = build_pipeline(preprocessor, model, version)

        best_model, best_params, best_score, training_time = train_model(
            pipeline,
            param_grid,
            X_train,
            y_train
        )

        print(f"done ({training_time:.1f}s, CV_MCC={best_score:.4f})")

        model_path = f"{MODELS_DIR}/{name}_{version}.pkl"
        joblib.dump(best_model, model_path)

        summary[name] = {
            "best_params": best_params,
            "best_cv_mcc": round(best_score, 4),
            "training_time_seconds": round(training_time, 2),
            "model_path": model_path
        }

    return summary


# Save training summary
def save_summary(full_summary, selected_summary, numeric_features, categorical_features):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {
        "full_features": full_summary,
        "selected_features": selected_summary,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "scoring": "matthews_corrcoef",
        "k_features": K_FEATURES,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_mapping": {
            "notckd": 0,
            "ckd": 1
        },
        "pipeline_note": (
            "Models are saved as full pipelines. Imputation, encoding, scaling, "
            "SMOTE, optional SelectKBest, and model training are handled inside "
            "GridSearchCV to reduce data leakage."
        )
    }

    with open(f"{RESULTS_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# Main pipeline
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading raw training data...")
    X_train, y_train = load_data()

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    full_summary = train_all_models("full", X_train, y_train, preprocessor)
    selected_summary = train_all_models("selected", X_train, y_train, preprocessor)

    save_summary(
        full_summary,
        selected_summary,
        numeric_features,
        categorical_features
    )

    print("\nTraining complete.")
    print("Saved 16 pipeline models to models/saved_models/")
    print("Saved training summary to results/training_summary.json")


if __name__ == "__main__":
    main()