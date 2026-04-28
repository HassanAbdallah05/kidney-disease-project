# Chronic Kidney Disease (CKD) Prediction Using Machine Learning

This project reproduces a CKD prediction pipeline based on the paper "Enhancing machine learning-based forecasting of chronic renal disease with explainable AI". The original work uses machine learning models, SMOTE, GridSearchCV, and SHAP/LIME explainability on the UCI CKD dataset.

Our extension adds SVM, XGBoost, SelectKBest feature selection, and MCC evaluation to improve the model comparison and provide a more balanced assessment.


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
-  `models/`: contains saved trained models.
- `notebooks/`: contains experiment notebooks.
- `requirements.txt` : contains required Python libraries


## References
- **Original paper:** [PeerJ Computer Science, vol. 10, e2291 (2024)](https://peerj.com/articles/cs-2291/)
- **Dataset:** [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)


## Team
- Hassan Abdalla 
- Mahmoud Abuzaanounah 

ML Course Project / Spring 2026
