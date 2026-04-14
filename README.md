# Immunotherapy-response-prediction
Overview

This project explores how machine learning can be used to understand and predict immunotherapy response using gene expression data.
Using the GSE91061 dataset, I built a predictive model and further analyzed the biological significance of key genes contributing to the model’s decisions.
Beyond model performance, the focus of this project is on extracting meaningful biological insights from high-dimensional data.

Objectives

Predict patient response to immunotherapy using gene expression profiles
Identify top 10 predictive genes driving model decisions

-Data Processing

Loaded and cleaned gene expression dataset (GSE91061).
Handled missing values and structured data for modeling. 

-Model Development

Trained a Random Forest Classifier.
Evaluated using accuracy, precision, recall, and F1-score.
Improved Bias using SMOTE. 

-Biological Interpretation

Extracted feature importance from the model.
Identified top 10 genes such as:
CXCL13
TIGIT
