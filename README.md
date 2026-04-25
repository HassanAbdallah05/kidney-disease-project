# kidney-disease-project

This project reproduces the paper 'Enhancing machine learning-based forecasting of chronic renal disease with explainable AI' and 
extends it by adding SVM, XGBoost, SelectKBest feature selection, and MCC for stronger evaluation.

## Extension
- Add SVM
- Add XGBoost
- Apply SelectKBest feature selection
- Compare results before and after feature selection
- Add MCC as an additional evaluation metric

## How to Run
1. install requirements:
   pip install -r requirements.txt

2. Run training and evaluation:
   python code/train_models.py

3. Results will be saved in:
   results/results_table.csv


# Code files
preprocess.py: clean data, impute missing values, encode, scale
feature_selection.py:  apply SelectKBest
train_models.py:  train baseline models + SVM + XGBoost
evaluate.py: compute accuracy, precision, recall, specificity, ROC-AUC, MCC
explainability.py: SHAP / LIME

## Folders
- data/: contains the CKD dataset
- code/: contains preprocessing, training, and evaluation code
- results/: contains output tables and figures
- models/: contains saved models
- notebooks/: contains experiment notebooks
