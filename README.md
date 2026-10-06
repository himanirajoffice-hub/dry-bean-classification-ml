# Dry Bean Type Classification using Machine Learning

## Project Overview

This project develops a supervised machine learning solution to classify dry beans into different varieties based on their physical and geometrical characteristics.

The dataset contains 16 numerical input features and seven bean classes:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

## Project Workflow

The project includes:

- Data loading and exploration
- Exploratory Data Analysis (EDA)
- Missing value and outlier analysis
- Feature preprocessing and scaling
- Stratified train-test splitting
- Training and comparison of multiple classification algorithms
- Cross-validation
- Class imbalance treatment using SMOTE
- Hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy, precision, recall, F1-score and confusion matrix
- Development of an interactive Streamlit prediction application

## Machine Learning Models Evaluated

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes
- Voting Classifier

## Final Model

SVM with SMOTE was selected as the final model.

Final model performance:

- Training Accuracy: 95.16%
- Test Accuracy: 92.43%
- Weighted F1-Score: 92.45%

The model demonstrated strong generalization performance on unseen test data.

## Streamlit Application

An interactive Streamlit application was developed to allow users to enter 16 physical and geometrical bean measurements and obtain the predicted bean class.

## Files

- `app.py` - Streamlit application
- `dry_bean_svm_model.pkl` - Trained SVM classification model

- ## 🚀 Live Application

The Dry Bean Type Classification model has been deployed as an interactive Streamlit web application.

**Live App:**  
https://dry-bean-classification-ml-c7pncnjntx8g9mctkptgxr.streamlit.app/

Users can enter the physical and geometrical measurements of a dry bean and obtain its predicted class using the trained SVM model.
- `dry_bean_scaler.pkl` - Fitted StandardScaler
- `Himani Supervised ML Classification Mini Project.ipynb` - Complete machine learning notebook
- `requirements.txt` - Required Python packages

## Author

**Himani Verma**
