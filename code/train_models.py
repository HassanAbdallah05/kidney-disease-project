import os
import json
import time
import warnings
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Configuration
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved_models"
RESULTS_DIR = "results"
RANDOM_STATE = 40
CV_FOLDS = 5


# Define models and hyperparameter grids
def get_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        ),

        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        ),

        "extra_trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20]
            }
        ),

        "lightgbm": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [-1, 10]
            }
        ),

        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "ann": (
            MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
            {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.001, 0.01]
            }
        ),

        "svm": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
        ),

        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6, 10]
            }
        )
    }


# Load data
def load_data(version):
    if version == "full":
        X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    else:
        X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_selected.csv")

    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").iloc[:, 0]

    return X_train, y_train


# Train one model
def train_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1
    )

    start_time = time.time()
    grid.fit(X_train, y_train)
    training_time = time.time() - start_time

    return grid.best_estimator_, grid.best_params_, grid.best_score_, training_time


# Train all models for one feature version
def train_all_models(version):
    print(f"\nTraining models using {version} features")

    X_train, y_train = load_data(version)
    models = get_models()
    summary = {}

    for name, (model, param_grid) in models.items():
        print(f"  Training {name}_{version}...", end=" ", flush=True)

        best_model, best_params, best_score, training_time = train_model(
            model,
            param_grid,
            X_train,
            y_train
        )

        print(f"done ({training_time:.1f}s, CV={best_score:.4f})")

        model_path = f"{MODELS_DIR}/{name}_{version}.pkl"
        joblib.dump(best_model, model_path)

        summary[name] = {
            "best_params": best_params,
            "best_cv_accuracy": round(best_score, 4),
            "training_time_seconds": round(training_time, 2),
            "model_path": model_path
        }

    return summary


# Save training summary
def save_summary(full_summary, selected_summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {
        "full_features": full_summary,
        "selected_features": selected_summary,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "scoring": "accuracy"
    }

    with open(f"{RESULTS_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# Main pipeline
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_summary = train_all_models("full")
    selected_summary = train_all_models("selected")

    save_summary(full_summary, selected_summary)

    print("\nTraining complete.")
    print("Saved 16 models to models/saved_models/")
    print("Saved training summary to results/training_summary.json")


if __name__ == "__main__":
    main()