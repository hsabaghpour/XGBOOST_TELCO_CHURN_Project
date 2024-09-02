# XGBOOST_TELCO_CHURN_Project

# Telecom Customer Churn Prediction

This project focuses on modeling customer churn for a telecom operator using machine learning techniques, particularly the XGBoost algorithm. The goal is to predict whether a customer will churn (leave the telecom service) based on various customer attributes.

## Project Overview

Customer churn is a critical problem in the telecom industry, where retaining existing customers is more cost-effective than acquiring new ones. This project leverages the XGBoost classifier, an efficient and scalable implementation of gradient boosting, to predict customer churn. The model was trained on a dataset containing various customer details such as contract type, payment method, and service usage patterns.

## Dataset

The dataset used for this project is `Telco_customer_churn.xlsx`, which includes information on customers who left the telecom company and those who stayed. The features were cleaned, preprocessed, and encoded to be suitable for machine learning modeling.

## Key Steps in the Project

1. **Data Preprocessing**:
   - Loaded the dataset and handled missing values.
   - Dropped unnecessary columns and replaced spaces in column names with underscores.
   - Converted categorical features into numerical values using one-hot encoding.
   - Corrected data types and ensured numerical values were properly formatted for modeling.

2. **Feature Engineering**:
   - Selected relevant features for the model.
   - Encoded categorical variables using one-hot encoding to make them suitable for the XGBoost model.

3. **Modeling with XGBoost**:
   - Split the dataset into training and testing sets.
   - Initialized and trained the `XGBClassifier` with an initial set of hyperparameters.
   - Evaluated the model using accuracy and ROC-AUC metrics.
   - Optimized the model by tuning hyperparameters to improve performance.

## Model Optimization

The XGBoost model was optimized by fine-tuning its hyperparameters. The following parameters were adjusted for better performance:

- **objective**: `'binary:logistic'`  
  Specifies the learning task as binary classification using a logistic regression model.

- **early_stopping_rounds**: `10`  
  Stops training if the validation score does not improve for 10 consecutive rounds.

- **eval_metric**: `'aucpr'`  
  Evaluation metric is set to the Area Under the Precision-Recall Curve, which is useful for imbalanced datasets.

- **gamma**: `0.25`  
  Minimum loss reduction required to make a further partition on a leaf node of the tree. A larger value leads to a more conservative model.

- **learning_rate**: `0.5`  
  Controls the step size at each iteration while moving toward a minimum of the loss function. A higher value accelerates training but risks overshooting.

- **max_depth**: `4`  
  Maximum depth of a tree. Controls model complexity and helps prevent overfitting.

- **n_estimators**: `20`  
  Number of boosting rounds or trees to build. Fewer trees can help reduce overfitting.

- **subsample**: `
