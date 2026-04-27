import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)


# Configuration
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved_models"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots/confusion_matrices"

POS_LABEL = 1   # CKD
NEG_LABEL = 0   # Not CKD

TOP_N_CONFUSION = 5

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "lightgbm",
    "decision_tree",
    "ann",
    "svm",
    "xgboost"
]

VERSIONS = ["full", "selected"]


# Load test data
def load_test_data(version):
    if version == "full":
        X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    else:
        X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test_selected.csv")

    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").iloc[:, 0]

    return X_test, y_test


# Get probability scores for ROC-AUC
def get_positive_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(POS_LABEL)
        return model.predict_proba(X_test)[:, class_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return model.predict(X_test)


# Evaluate one model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = get_positive_scores(model, X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[NEG_LABEL, POS_LABEL]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_score),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred


# Evaluate all models
def evaluate_all_models():
    results = []
    predictions = {}

    for version in VERSIONS:
        X_test, y_test = load_test_data(version)

        for name in MODEL_NAMES:
            model_path = f"{MODELS_DIR}/{name}_{version}.pkl"

            if not os.path.exists(model_path):
                print(f"Skipping missing model: {model_path}")
                continue

            model = joblib.load(model_path)
            metrics, y_pred = evaluate_model(model, X_test, y_test)

            row = {
                "Model": name,
                "Version": version
            }

            row.update({key: round(value, 4) for key, value in metrics.items()})
            results.append(row)

            predictions[f"{name}_{version}"] = (y_test, y_pred)

    return pd.DataFrame(results), predictions


# Plot one confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_label):
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[NEG_LABEL, POS_LABEL]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix — {model_label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No CKD", "CKD"])
    ax.set_yticklabels(["No CKD", "CKD"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{model_label}.png", dpi=150, bbox_inches="tight")
    plt.close()


# Plot top N confusion matrices
def plot_top_confusion_matrices(results_df, predictions):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    top_models = results_df.head(TOP_N_CONFUSION)

    for _, row in top_models.iterrows():
        key = f"{row['Model']}_{row['Version']}"
        y_test, y_pred = predictions[key]
        plot_confusion_matrix(y_test, y_pred, key)


# Save result tables
def save_results(results_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df.to_csv(f"{RESULTS_DIR}/results_table.csv", index=False)

    full_df = results_df[results_df["Version"] == "full"].reset_index(drop=True)
    selected_df = results_df[results_df["Version"] == "selected"].reset_index(drop=True)

    full_df.to_csv(f"{RESULTS_DIR}/results_without_fs.csv", index=False)
    selected_df.to_csv(f"{RESULTS_DIR}/results_with_fs.csv", index=False)


# Print summary
def print_summary(results_df):
    print("\nAll models sorted by MCC:")
    print(results_df.to_string(index=False))

    print(f"\nTop {TOP_N_CONFUSION} models:")
    print(results_df.head(TOP_N_CONFUSION).to_string(index=False))


# Main pipeline
def main():
    results_df, predictions = evaluate_all_models()

    results_df = results_df.sort_values("MCC", ascending=False).reset_index(drop=True)

    save_results(results_df)
    plot_top_confusion_matrices(results_df, predictions)
    print_summary(results_df)

    print("\nEvaluation complete.")
    print("Results saved to results/")
    print(f"Top {TOP_N_CONFUSION} confusion matrices saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()