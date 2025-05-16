# Credit Risk Prediction – Give Me Some Credit (Kaggle Competition)

This repository contains the work on the [**Give Me Some Credit**](https://www.kaggle.com/competitions/GiveMeSomeCredit) Kaggle competition, which focuses on predicting the probability that a borrower will experience financial distress in the next two years.

We explore various data preprocessing strategies and machine learning models to build a robust credit risk prediction system.

---

## 🔍 Problem Overview

The task is to perform binary classification, where the goal is to predict whether an individual is likely to default on a loan. This dataset presents challenges such as **missing values**, **outliers**, and **class imbalance**, all of which are addressed through structured experiments.

---

## 📊 Dataset

- **Source:** [Kaggle Competition - Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- **Data Type:** Tabular, numeric
- **Target Variable:** `SeriousDlqin2yrs` (1 = default, 0 = non-default)
- **Features:** Income, debt ratio, number of dependents, late payments, etc.

---

## 🧪 Experiments

We designed two core experiments to progressively improve model performance while addressing data quality and class imbalance issues.

---

### ✅ Experiment 01 – Baseline

A simple baseline approach to establish initial model performance.

#### 🔧 Key Steps:
- **Missing Value Handling:** Dropped all rows with missing values from both train and test sets.
- **Outlier Treatment:** Applied IQR-based capping **only on the training set** to prevent test set leakage.

#### 🤖 Models Evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- AdaBoost
- Artificial Neural Network (ANN)

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.931               | 0.933         | 0.932   | 0.933    | 0.934    | 0.933  |
| Precision    | 0.747               | 0.744         | 0.737   | 0.753    | 0.751    | 0.746  |
| Recall       | 0.525               | 0.584         | 0.593   | 0.583    | 0.597    | 0.583  |
| F1 Score     | 0.530               | 0.617         | 0.627   | 0.617    | 0.633    | 0.616  |
| AUC-ROC      | 0.792               | 0.825         | 0.843   | 0.852    | 0.848    | 0.852  |

#### ✅ Takeaways:
- Good baseline performance.
- No SMOTE or imputation – simple but clean setup.
- Test set remained untouched to simulate a real unseen environment.

---

### ⚙️ Experiment 02 – Imputation & Imbalance Handling

Builds upon the baseline by intelligently imputing missing values and balancing classes.

#### 🔧 Key Steps:
- **KNN Imputation:** Missing values in both train and test were imputed using K-Nearest Neighbors. The imputer was fit only on the training data.
- **IQR Outlier Capping:** Same as Experiment 01, but applied *only to the training set*.
- **SMOTE Oversampling:** Applied to the training set only to address class imbalance.

#### 🤖 Models Evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- AdaBoost
- Artificial Neural Network (ANN)

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.764               | 0.932         | 0.935   | 0.935    | 0.917    | 0.906  |
| Precision    | 0.580               | 0.716         | 0.739   | 0.744    | 0.677    | 0.654  |
| Recall       | 0.750               | 0.610         | 0.600   | 0.610    | 0.700    | 0.699  |
| F1 Score     | 0.576               | 0.642         | 0.635   | 0.647    | 0.687    | 0.672  |
| AUC-ROC      | 0.829               | 0.833         | 0.854   | 0.861    | 0.842    | 0.836  |

#### ✅ Takeaways:
- SMOTE improves recall significantly at a slight cost to precision.
- KNN imputation helps retain more data compared to row dropping.
- AdaBoost and ANN emerge as top-performing models.
