import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


# Configuration
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved_models"
PLOTS_DIR = "results/plots/explainability"

MODEL_NAME = "xgboost_full"
MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}.pkl"
CLASS_NAMES = ["No CKD", "CKD"]
POS_LABEL = 1


# Load data and model
def load_data_and_model():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").iloc[:, 0]
    model = joblib.load(MODEL_PATH)
    return model, X_train, X_test, y_test


# Pick the first CKD patient for explanation
def choose_patient_index(y_test):
    ckd_indices = y_test[y_test == POS_LABEL].index.tolist()
    return ckd_indices[0] if ckd_indices else 0


# Get SHAP values
def get_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = explainer.expected_value[1]
    else:
        expected_value = explainer.expected_value

    return shap_values, expected_value


# Create SHAP plots
def run_shap(model, X_test, patient_index):
    shap_values, expected_value = get_shap_values(model, X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    patient_position = list(X_test.index).index(patient_index)

    explanation = shap.Explanation(
        values=shap_values[patient_position],
        base_values=expected_value,
        data=X_test.loc[patient_index].values,
        feature_names=X_test.columns.tolist()
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


# Create LIME explanation
def run_lime(model, X_train, X_test, patient_index):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True
    )

    patient = X_test.loc[patient_index].values

    explanation = explainer.explain_instance(
        data_row=patient,
        predict_fn=model.predict_proba,
        num_features=10
    )

    explanation.save_to_file(f"{PLOTS_DIR}/lime_explanation.html")

    fig = explanation.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/lime_explanation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Main pipeline
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model, X_train, X_test, y_test = load_data_and_model()
    patient_index = choose_patient_index(y_test)

    print(f"Running SHAP for {MODEL_NAME}...")
    run_shap(model, X_test, patient_index)

    print(f"Running LIME for patient index {patient_index}...")
    run_lime(model, X_train, X_test, patient_index)

    print("Explainability complete.")
    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()