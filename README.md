````markdown
# Chronic Kidney Disease (CKD) Prediction Using Machine Learning

This project reproduces and extends a CKD prediction pipeline based on the paper **"Enhancing machine learning-based forecasting of chronic renal disease with explainable AI"**. The original work uses machine learning models, SMOTE, GridSearchCV, and SHAP/LIME explainability on the UCI CKD dataset.

Our extension focuses on making the model comparison broader and the evaluation more reliable by adding **SVM**, **SelectKBest feature selection**, **MCC**, and **5-fold cross-validation**.

---

## Our Extensions

1. Add **SVM** as an additional classifier to the original model comparison.
2. Apply **SelectKBest** feature selection and compare full features with selected features.
3. Add **MCC** as a balanced evaluation metric for binary medical classification.
4. Use **5-fold cross-validation** to make the evaluation more reliable on the small CKD dataset.
5. Use **SHAP and LIME** to explain model predictions.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/HassanAbdallah05/kidney-disease-project.git
cd kidney-disease-project
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python code/preprocess.py
python code/train_models.py
python code/evaluate.py
python code/cross_validate.py
python code/explainability.py
```

Optional:

```bash
python code/feature_selection.py
```

---

## Code Files

* `preprocess.py`: cleans the raw data and creates train/test files.
* `train_models.py`: trains machine learning models and saves the trained pipelines.
* `evaluate.py`: evaluates saved models on the held-out test set.
* `cross_validate.py`: runs the main 5-fold cross-validation evaluation.
* `explainability.py`: generates SHAP and LIME explanations.
* `feature_selection.py`: optional script for SelectKBest feature-ranking output.

---

## Folders

* `data/`: contains the CKD dataset and processed files.
* `code/`: contains preprocessing, training, evaluation, cross-validation, and explainability scripts.
* `results/`: contains output tables and figures.
* `models/`: contains saved trained models.
* `notebooks/`: contains experiment notebooks.
* `requirements.txt`: contains required Python libraries.

---

## Final Results

The main final evaluation uses **5-fold cross-validation**. The table below reports mean scores across the folds.

| Model               | Accuracy | ROC-AUC |    MCC |
| ------------------- | -------: | ------: | -----: |
| Random Forest       |   0.9900 |  0.9999 | 0.9789 |
| Logistic Regression |   0.9875 |  0.9996 | 0.9738 |
| LightGBM            |   0.9875 |  0.9997 | 0.9736 |
| ANN                 |   0.9850 |  0.9996 | 0.9689 |
| SVM                 |   0.9775 |  0.9984 | 0.9525 |

Random Forest achieved the best full-feature performance. SVM was added as an extension model and achieved competitive performance.

### Feature Selection Results

SelectKBest was tested with `k = 10` features. The table below compares MCC before and after feature selection.

| Model         | Full MCC | SelectKBest MCC |
| ------------- | -------: | --------------: |
| Extra Trees   |   0.9628 |          0.9842 |
| SVM           |   0.9525 |          0.9792 |
| LightGBM      |   0.9736 |          0.9792 |
| Random Forest |   0.9789 |          0.9792 |

SelectKBest improved some models, especially SVM and Extra Trees. This shows that feature selection can reduce the number of clinical features while maintaining strong performance.

---

## Notes

The UCI CKD dataset is small, with only 400 patient records. Some clinical features are highly predictive, so high scores should be interpreted carefully. Future work should test the pipeline on larger and more diverse hospital datasets.

---

## References

* **Original paper:** [PeerJ Computer Science, vol. 10, e2291 (2024)](https://peerj.com/articles/cs-2291/)
* **Dataset:** [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

---

## Team

* Hassan Abdalla
* Mahmoud Abuzaanounah

ML Course Project / Spring 2026

```
```
