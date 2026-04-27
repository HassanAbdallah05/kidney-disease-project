import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import joblib


# Configuration
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
K_FEATURES = 10


# Load preprocessed data
def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").iloc[:, 0]
    return X_train, X_test, y_train


# Apply SelectKBest with ANOVA F-value
def apply_selectkbest(X_train, y_train, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    ranking_df = pd.DataFrame({
        "Feature": X_train.columns.tolist(),
        "F_Score": selector.scores_,
        "P_Value": selector.pvalues_,
        "Selected": selector.get_support()
    }).sort_values("F_Score", ascending=False).reset_index(drop=True)

    return selector, ranking_df


# Transform train and test sets
def transform_data(selector, X_train, X_test):
    selected_features = X_train.columns[selector.get_support()].tolist()

    X_train_selected = pd.DataFrame(
        selector.transform(X_train),
        columns=selected_features,
        index=X_train.index
    )

    X_test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=selected_features,
        index=X_test.index
    )

    return X_train_selected, X_test_selected, selected_features


# Plot feature importance
def plot_feature_ranking(ranking_df, k):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    colors = ["#2ecc71" if selected else "#bdc3c7" for selected in ranking_df["Selected"]]

    plt.figure(figsize=(10, 8))
    plt.barh(
        ranking_df["Feature"][::-1],
        ranking_df["F_Score"][::-1],
        color=colors[::-1]
    )
    plt.xlabel("ANOVA F-Score")
    plt.title(f"Feature Importance — Top {k} Selected (Green)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


#save outputs
def save_outputs(X_train_selected, X_test_selected, selector, ranking_df, selected_features):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train_selected.to_csv(f"{PROCESSED_DIR}/X_train_selected.csv", index=False)
    X_test_selected.to_csv(f"{PROCESSED_DIR}/X_test_selected.csv", index=False)

    joblib.dump(selector, f"{PROCESSED_DIR}/selector.pkl")
    ranking_df.to_csv(f"{RESULTS_DIR}/feature_ranking.csv", index=False)

    metadata_path = f"{PROCESSED_DIR}/metadata.json"

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["k_features"] = K_FEATURES
    metadata["selected_features"] = selected_features
    metadata["feature_selection_method"] = "SelectKBest (ANOVA F-value)"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)




# Main pipeline
def main():
    X_train, X_test, y_train = load_data()

    selector, ranking_df = apply_selectkbest(X_train, y_train, K_FEATURES)

    X_train_selected, X_test_selected, selected_features = transform_data(
        selector,
        X_train,
        X_test
    )

    plot_feature_ranking(ranking_df, K_FEATURES)

    save_outputs(
        X_train_selected,
        X_test_selected,
        selector,
        ranking_df,
        selected_features
    )

    print(f"Feature selection complete. Top {K_FEATURES} features: {selected_features}")


if __name__ == "__main__":
    main()