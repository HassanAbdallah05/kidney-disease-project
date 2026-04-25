# Chronic Kidney Disease (CKD) Prediction Using Machine Learning

This project reproduces the paper 'Enhancing machine learning-based forecasting of chronic renal disease with explainable AI' and 
extends it by adding SVM, XGBoost, SelectKBest feature selection, and MCC for stronger evaluation.

## Extension
- Add SVM
- Add XGBoost
- Apply SelectKBest feature selection
- Compare results before and after feature selection
- Add MCC as an additional evaluation metric

Our Extensions: 
1. **Add SVM and XGBoost** as two additional classifiers to the original model comparison.
2. **Apply SelectKBest** (ANOVA F-value) for feature selection, and compare model performance **before and after** feature selection.
3. **Add MCC and Cohen's Kappa** as additional evaluation metrics, which are more reliable for imbalanced medical data.


## How to Run
### 1. Clone the repository
```bash
git clone https://github.com/HassanAbdallah05/kidney-disease-project.git
cd kidney-disease-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python code/train_models.py
```



## Code files
- preprocess.py: used for clean data, impute missing values, encode, scale
- feature_selection.py: used for apply SelectKBest
- train_models.py: used train baseline models + SVM + XGBoost
- evaluate.py: compute accuracy, precision, recall, specificity, ROC-AUC, MCC
- explainability.py: used SHAP / LIME


## Folders
- `data/` : contains the CKD dataset
- `code/` : contains preprocessing, training, evaluation, and explainability scripts
- `results/` : contains output tables and figures
- `models/:` contains saved models
- `notebooks/:` contains experiment notebooks
- `requirements.txt` : contains required Python libraries


