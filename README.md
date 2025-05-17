# Credit Risk Prediction – Give Me Some Credit (Kaggle Competition)

This repository contains our work on the [**Give Me Some Credit**](https://www.kaggle.com/competitions/GiveMeSomeCredit) Kaggle competition, which focuses on predicting the probability that a borrower will experience financial distress in the next two years.

We explore a variety of data preprocessing techniques and machine learning models to build a robust credit risk prediction pipeline, carefully handling challenges such as **missing data**, **outliers**, and **class imbalance**.

---

## 🔍 Problem Overview

The objective is to perform **binary classification**: predicting whether an individual will default (`1`) or not (`0`) on a loan within two years. The dataset presents real-world issues such as class imbalance, missing values, and noisy data.

---

## 📊 Dataset

- **Source:** [Kaggle - Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- **Data Type:** Tabular, numerical
- **Target Variable:** `SeriousDlqin2yrs` (1 = default, 0 = non-default)
- **Features:** Includes income, debt ratio, age, number of dependents, late payment history, and more.

---

## 🧪 Experiments

Our pipeline consists of three major experiments that progressively enhance performance by introducing imputation, class balancing, and feature selection.

---

### ✅ Experiment 01 – Baseline

A minimal setup to establish a performance benchmark.

#### 🔧 Key Steps:
- **Missing Values:** Dropped rows with missing values.
- **Outliers:** Applied IQR-based capping **only on the training set** to prevent data leakage.

#### 🤖 Models Evaluated:
Logistic Regression, Random Forest, XGBoost, LightGBM, AdaBoost, ANN

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.931               | 0.933         | 0.932   | 0.933    | 0.934    | 0.933  |
| Precision    | 0.747               | 0.744         | 0.737   | 0.753    | 0.751    | 0.746  |
| Recall       | 0.525               | 0.584         | 0.593   | 0.583    | 0.597    | 0.583  |
| F1 Score     | 0.530               | 0.617         | 0.627   | 0.617    | 0.633    | 0.616  |
| AUC-ROC      | 0.792               | 0.825         | 0.843   | 0.852    | 0.848    | 0.852  |

#### ✅ Takeaways:
- Simple yet effective pipeline.
- No imputation or SMOTE, but good initial results.
- Ensured fair evaluation by avoiding leakage.

---

### ⚙️ Experiment 02 – Imputation & Class Balancing Variants

We improved upon the baseline by imputing missing values and applying different class balancing strategies. This experiment included comparative analysis of **SMOTE**, **Tomek Links**, and **SMOTETomek**.

#### 🔧 Key Steps:
- **Missing Value Imputation:** KNN Imputer (fit on training set only).
- **Outlier Treatment:** IQR capping (training set only).
- **Class Balancing:** Compared 3 techniques: SMOTE, Tomek Links, and SMOTETomek.

#### 🧪 SMOTE Results:

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.764               | 0.933         | 0.934   | 0.935    | 0.918    | 0.902  |
| Precision    | 0.580               | 0.721         | 0.735   | 0.742    | 0.679    | 0.656  |
| Recall       | 0.750               | 0.609         | 0.599   | 0.612    | 0.696    | 0.729  |
| F1 Score     | 0.576               | 0.642         | 0.634   | 0.649    | 0.687    | 0.683  |
| AUC-ROC      | 0.829               | 0.832         | 0.855   | 0.861    | 0.837    | 0.846  |

#### 🧪 Tomek Links Results:

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.934               | 0.935         | 0.935   | 0.937    | 0.936    | 0.937  |
| Precision    | 0.763               | 0.746         | 0.742   | 0.765    | 0.753    | 0.773  |
| Recall       | 0.527               | 0.593         | 0.601   | 0.595    | 0.606    | 0.592  |
| F1 Score     | 0.535               | 0.628         | 0.636   | 0.633    | 0.644    | 0.629  |
| AUC-ROC      | 0.808               | 0.841         | 0.855   | 0.864    | 0.860    | 0.865  |

#### 🧪 SMOTETomek Results:

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|--------|
| Accuracy     | 0.768               | 0.933         | 0.934   | 0.935    | 0.918    | 0.901  |
| Precision    | 0.581               | 0.719         | 0.733   | 0.743    | 0.678    | 0.651  |
| Recall       | 0.753               | 0.612         | 0.600   | 0.614    | 0.692    | 0.718  |
| F1 Score     | 0.580               | 0.644         | 0.635   | 0.650    | 0.685    | 0.676  |
| AUC-ROC      | 0.833               | 0.833         | 0.856   | 0.862    | 0.838    | 0.847  |

#### ✅ Takeaways:
- All three methods improved performance compared to the baseline.
- **SMOTETomek** consistently provided the best balance between **precision**, **recall**, and **AUC**, and was selected for further experiments.

---

### 🧪 Experiment 03 – Feature Selection + SMOTETomek

In this final experiment, we incorporated feature selection and applied **SMOTETomek** as the class balancing technique, based on insights from Experiment 02.

---

#### 🔧 Key Steps:
- **Missing Value Imputation:** Used KNN Imputer, fitted only on the training set to prevent data leakage.
- **Outlier Treatment:** Applied IQR-based capping on the training data to handle extreme values.
- **Class Balancing:**  Applied **SMOTETomek** as the class balancing technique.
- **Feature Selection:**
  - Initially applied **Recursive Feature Elimination (RFE)** with **Random Forest** and **GridSearchCV** — however, it retained all features, suggesting no strong preference for elimination.
  - Subsequently used **Permutation Importance (Feature Shuffle variant)** with **AdaBoost**, which was the best performing model at that stage. This revealed 8 key features used for the final experiment.


- `NumberOfTimes90DaysLate`
- `RevolvingUtilizationOfUnsecuredLines`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `age`
- `NumberRealEstateLoansOrLines`
- `NumberOfDependents`
- `DebtRatio`

These features were then used for training with **SMOTETomek** to assess their impact on model performance.

---

### 📊 Final Performance with Selected 8 Features

| Metric       | Logistic Regression | Random Forest | XGBoost | LightGBM | AdaBoost | ANN   |
|--------------|---------------------|---------------|---------|----------|----------|-------|
| **Accuracy** | 0.770               | 0.927         | 0.934   | 0.934    | 0.918    | 0.904 |
| **Precision**| 0.581               | 0.688         | 0.727   | 0.733    | 0.678    | 0.654 |
| **Recall**   | 0.749               | 0.622         | 0.606   | 0.617    | 0.692    | 0.712 |
| **F1 Score** | 0.579               | 0.646         | 0.639   | 0.652    | 0.685    | 0.676 |
| **AUC-ROC**  | 0.830               | 0.820         | 0.848   | 0.855    | 0.838    | 0.842 |

---

### ✅ Conclusion

- **SMOTETomek** was found to be the most effective class balancing technique.
- **Permutation Importance** provided more meaningful and targeted feature selection than RFE.
- Performance was noted to be **consistentl with the last experiment 02 (SMOTETomek)** using the selected 8 features.
- **LightGBM**, **AdaBoost**, and **ANN** stood out with better f1, recall and AUC-ROC values.


---

## 🚀 Future Work

- Apply advanced techniques like ensemble stacking or blending.
- Explore explainability (e.g., SHAP, LIME) to interpret model decisions.
- Evaluate temporal aspects if longitudinal data is available.

---

