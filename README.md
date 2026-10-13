# Telco Customer Churn Prediction & Feature Engineering Pipeline

An end-to-end Machine Learning pipeline built to predict customer churn for a telecommunications company. This repository covers everything from exploratory data analysis (EDA) and robust feature engineering to hyperparameter tuning and model evaluation across multiple algorithms.

---

## 📌 Project Overview
Customer churn is one of the most critical metrics for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones. 
This project analyzes demographic info, account details, and service subscriptions of **7,043 customers** to identify patterns, engineer high-impact features, and build classification models that accurately predict whether a customer will leave the service.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.14.4
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost
* **Environment Management:** Pip, venv

### Installed Package Versions
catboost          1.2.10
cloudpickle       3.1.2
contourpy         1.4.0
joblib            1.6.0
lightgbm          4.7.0
matplotlib        3.11.2
numpy             2.5.4
pandas            3.0.6
scikit-learn      1.9.1
scipy             1.18.1
seaborn           0.13.2
xgboost           3.4.1

---

## Project Structure
├── Telco-Customer-Churn.csv         # Raw dataset
├── Telco_Customer_Churn.py          # Main script containing EDA, Feature Engineering & Modeling
└── README.md                        # Project documentation

## Pipeline Steps
# 1. Exploratory Data Analysis (EDA)
General Inspection: Checked data shapes, data types, missing values, and quantile distributions.

Data Type Correction: Converted TotalCharges into a numeric format and encoded the Churn target variable (Yes: 1, No: 0).

Categorical & Numerical Analysis: Summarized distributions, analyzed relationships with the target variable (Churn), and examined core correlations (e.g., strong relationships between tenure, MonthlyCharges, and TotalCharges).

# 2. Feature Engineering
Created domain-specific features to boost model predictive power:

NEW_TENURE_YEAR: Grouped customers into yearly tenure brackets (0-1 Year, 1-2 Year, etc.).

NEW_Engaged: Flagged customers with 1- or 2-year contracts.

NEW_noProt: Identified customers lacking online backup, device protection, or tech support.

NEW_Young_Not_Engaged: Highlighted non-senior customers on month-to-month contracts.

NEW_TotalServices: Counted the total number of subscribed services per customer.

NEW_FLAG_ANY_STREAMING: Flagged users utilizing TV or movie streaming services.

NEW_FLAG_AutoPayment: Identified automated payment methods (bank transfers or credit cards).

NEW_AVG_Charges & NEW_Increase: Derived average charges per tenure month and analyzed price escalation ratios.

NEW_AVG_Service_Fee: Calculated average cost per active service subscribed.

# 3. Preprocessing & Encoding
Outlier Treatment: Handled extreme values using custom threshold functions based on custom quantiles.

Encoding: Applied Label Encoding on binary categorical columns and One-Hot Encoding on multi-class categorical variables.

# 4. Model Training & Evaluation
Trained and evaluated multiple machine learning classifiers using 10-fold cross-validation, measuring Accuracy, F1-Score, ROC-AUC, Precision, and Recall:

Logistic Regression

K-Nearest Neighbors (KNN)

CART (Decision Trees)

Random Forest

Support Vector Machine (SVM)

XGBoost

LightGBM

CatBoost

Hyperparameter tuning via GridSearchCV was performed for tree-based ensemble models (Random Forest, XGBoost, LightGBM, and CatBoost) to optimize final performance. Feature importance plots were generated to interpret driving factors behind customer churn.

## Clone Repository
git clone [https://github.com/dudukbb/Telco_Customer_Churn_Prediction.git](https://github.com/dudukbb/Telco_Customer_Churn_Prediction.git)
cd Telco_Customer_Churn_Prediction
python Telco_Customer_Churn.py

