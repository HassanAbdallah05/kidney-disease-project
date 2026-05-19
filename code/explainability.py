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

# Use SVM for presentation
MODEL_NAME = "svm_full"
MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}.pkl"

CLASS_NAMES = ["No CKD", "CKD"]
POS_LABEL = 1

NUM_LIME_FEATURES = 5
BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 50
RANDOM_STATE = 40


# Load raw data and saved pipeline model
def load_data_and_model():
    X_train_raw = pd.read_csv(f"{PROCESSED_DIR}/X_train_raw.csv")
    X_test_raw = pd.read_csv(f"{PROCESSED_DIR}/X_test_raw.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").iloc[:, 0]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run python code/train_models.py first."
        )

    model_pipeline = joblib.load(MODEL_PATH)

    return model_pipeline, X_train_raw, X_test_raw, y_test


# Pick first CKD patient
def choose_patient_position(y_test):
    ckd_positions = y_test[y_test == POS_LABEL].index.tolist()
    return ckd_positions[0] if ckd_positions else 0


# Clean transformed feature names
def clean_feature_names(feature_names):
    cleaned = []

    for name in feature_names:
        name = str(name)
        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("remainder__", "")
        cleaned.append(name)

    return cleaned


# Transform raw data using saved pipeline preprocessing
def transform_data_for_explainability(model_pipeline, X_train_raw, X_test_raw):
    preprocessor = model_pipeline.named_steps["preprocess"]

    X_train_transformed = preprocessor.transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    feature_names = clean_feature_names(preprocessor.get_feature_names_out())

    if "feature_selection" in model_pipeline.named_steps:
        selector = model_pipeline.named_steps["feature_selection"]

        X_train_transformed = selector.transform(X_train_transformed)
        X_test_transformed = selector.transform(X_test_transformed)

        selected_mask = selector.get_support()
        feature_names = [
            name for name, keep in zip(feature_names, selected_mask) if keep
        ]

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    final_model = model_pipeline.named_steps["model"]

    return final_model, X_train_df, X_test_df


# Predict CKD probability for SHAP KernelExplainer
def predict_ckd_probability(final_model, X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    probabilities = final_model.predict_proba(X)
    class_index = list(final_model.classes_).index(POS_LABEL)

    return probabilities[:, class_index]


# SHAP for SVM using KernelExplainer
def run_shap_svm(final_model, X_train, X_test, patient_position):
    print("Preparing SHAP background sample...")

    background = shap.sample(
        X_train,
        min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE
    )

    X_explain = X_test.iloc[:min(EXPLAIN_SIZE, len(X_test))].copy()

    if patient_position >= len(X_explain):
        patient_row = X_test.iloc[[patient_position]]
        X_explain = pd.concat([X_explain, patient_row], axis=0)

    print("Running SHAP KernelExplainer for SVM...")

    explainer = shap.KernelExplainer(
        lambda data: predict_ckd_probability(
            final_model,
            pd.DataFrame(data, columns=X_train.columns)
        ),
        background
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=100
    )

    expected_value = explainer.expected_value

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/svm_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/svm_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP waterfall plot
    if patient_position < len(X_explain):
        explanation_position = patient_position
    else:
        explanation_position = len(X_explain) - 1

    explanation = shap.Explanation(
        values=np.array(shap_values)[explanation_position],
        base_values=expected_value,
        data=X_explain.iloc[explanation_position].values,
        feature_names=X_explain.columns.tolist()
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/svm_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


# LIME explanation for SVM
def run_lime(final_model, X_train, X_test, patient_position):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True
    )

    patient = X_test.iloc[patient_position].values

    explanation = explainer.explain_instance(
        data_row=patient,
        predict_fn=final_model.predict_proba,
        num_features=NUM_LIME_FEATURES
    )

    explanation.save_to_file(f"{PLOTS_DIR}/svm_lime_explanation.html")

    fig = explanation.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/svm_lime_explanation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Main pipeline
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_pipeline, X_train_raw, X_test_raw, y_test = load_data_and_model()

    final_model, X_train, X_test = transform_data_for_explainability(
        model_pipeline,
        X_train_raw,
        X_test_raw
    )

    patient_position = choose_patient_position(y_test)

    print(f"Running explainability for {MODEL_NAME}...")
    print(f"Selected patient position: {patient_position}")

    run_shap_svm(final_model, X_train, X_test, patient_position)
    run_lime(final_model, X_train, X_test, patient_position)

    print("Explainability complete.")
    print(f"Plots saved to {PLOTS_DIR}/")
    print("Generated files:")
    print("  - svm_shap_summary.png")
    print("  - svm_shap_bar.png")
    print("  - svm_shap_waterfall.png")
    print("  - svm_lime_explanation.png")
    print("  - svm_lime_explanation.html")


if __name__ == "__main__":
    main()