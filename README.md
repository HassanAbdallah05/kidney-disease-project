# kidney-disease-project

This project reproduces the paper 'Enhancing machine learning-based forecasting of chronic renal disease with explainable AI' and 
extends it by adding SVM, XGBoost, SelectKBest feature selection, and MCC for stronger evaluation.


# Code files:
preprocess.py: clean data, impute missing values, encode, scale
feature_selection.py:  apply SelectKBest
train_models.py:  train baseline models + SVM + XGBoost
evaluate.py: compute accuracy, precision, recall, specificity, ROC-AUC, MCC
explainability.py: SHAP / LIME
