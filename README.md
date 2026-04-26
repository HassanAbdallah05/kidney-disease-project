# Chronic Kidney Disease (CKD) Prediction Using Machine Learning

The original paper builds an end-to-end CKD prediction pipeline using six ML models (Logistic Regression, Random Forest, Extra Trees, LightGBM, Decision Tree, ANN), with GridSearchCV tuning, SMOTE class balancing, and SHAP/LIME explainability achieving 99.07% accuracy on the UCI dataset.
This project **reproduces** that work and **extends** it in three ways.


## Our Extensions: 
1. Add SVM and XGBoost as two additional classifiers to the original model comparison.
2. Apply SelectKBest for feature selection, and compare model performance **before and after** feature selection.
3. Add MCC as an additional evaluation metric to provide a more balanced assessment for imbalanced medical data.


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
python code/preprocess.py
python code/feature_selection.py
python code/train_models.py
python code/evaluate.py
python code/explainability.py
```



## Code files
- preprocess.py: cleans data, impute missing values, encode, scale
- feature_selection.py: used for apply SelectKBest
- train_models.py: trains baseline models + SVM + XGBoost
- evaluate.py: compute accuracy, precision, recall, specificity, ROC-AUC, MCC
- explainability.py: used SHAP / LIME


## Folders
- `data/` : contains the CKD dataset
- `code/` : contains preprocessing, training, evaluation, and explainability scripts
- `results/` : contains output tables and figures
- `models/:` contains saved models
- `notebooks/:` contains experiment notebooks
- `requirements.txt` : contains required Python libraries


## References
- **Original Paper:** [PeerJ Computer Science, vol. 10, e2291 (2024)](https://peerj.com/articles/cs-2291/)
- **Dataset:** [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)


## Teams
- Hassan Abdalla 
- Mahmoud Abuzaanounah 

ML Course Project — Spring 2026
