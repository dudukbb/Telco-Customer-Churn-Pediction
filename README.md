# 📉 Telco Customer Churn Prediction 

An end-to-end Machine Learning project designed to predict customer churn for a telecommunications company, featuring comprehensive exploratory data analysis (EDA), advanced feature engineering, hyperparameter optimization, and automated output generation.

---

## 🔗 Project Overview & Scope
This project covers the entire machine learning lifecycle: from raw data inspection and exploratory data analysis to feature engineering, baseline benchmarking, hyperparameter tuning, and automated tracking.

### 🔗 Key Pipeline Steps:
1. **Exploratory Data Analysis (EDA):** Inspected data structure, analyzed categorical and numerical variables, and visualized relationships against the target variable (`Churn`).
2. **Missing & Outlier Value Analysis:** Handled missing values with appropriate imputation and suppressed outliers based on defined statistical thresholds.
3. **Feature Engineering:** Created domain-specific features to capture customer behavior patterns and boost predictive power.
4. **Modeling & Hyperparameter Tuning:** Optimized high-performance algorithms (`RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`) using `GridSearchCV` and k-fold cross-validation.
5. **Automated Output Management:** Automatically saved all performance metrics tables (`.csv` & `.png`) and feature importance charts (`.png`) into a structured `outputs/` directory.

---

## 🔗 Engineered Features Created
Domain-specific features engineered to capture customer loyalty and behavioral trends:
* **`NEW_TENURE_YEAR`**: Grouping customer tenure into yearly bins (0-1 year up to 5-6 years).
* **`NEW_Engaged`**: Binary flag indicating whether the customer has a long-term contract (One year or Two year).
* **`NEW_noProt`**: Indicator for customers lacking essential online security, backup, or tech support services.
* **`NEW_TotalServices`**: Total count of active services subscribed by the customer.
* **`NEW_FLAG_ANY_STREAMING`**: Flag indicating usage of either TV or movie streaming services.
* **`NEW_FLAG_AutoPayment`**: Indicator for automated payment methods (Credit card or Bank transfer).
* **`NEW_AVG_Charges` & `NEW_Increase`**: Financial metrics tracking customer charge behaviors over tenure.
* **`NEW_AVG_Service_Fee`**: Average fee paid per subscribed service.

---

## 🔗 Tech Stack & Libraries
* **Python** 
* **Data Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning & Modeling:** 
  * `Scikit-Learn` (`GridSearchCV`, `cross_validate`, `LabelEncoder`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`)
  * `XGBoost`
  * `LightGBM`
  * `CatBoost`

---

## 📂 Project Directory Structure
```text
Telco_Customer_Churn_Prediction/
│
├── outputs/
│   ├── eda/                                   # EDA distribution and target analysis plots
│   │   ├── categorical_plots/
│   │   ├── numerical_plots/
│   │   ├── target_analysis_categoric/
│   │   ├── target_analysis_categoric_plots/
│   │   ├── target_analysis_numeric/
│   │   └── target_analysis_numeric_plots/
│   ├── importances/                           # Feature importance visual charts
│   │   ├── RandomForest_importance.png
│   │   ├── XGBoost_importance.png
│   │   ├── LightGBM_importance.png
│   │   └── CatBoost_importance.png
│   └── metrics/                               # Performance summary tables
│       ├── base_models_summary.png
│       └── engineered_models_summary.png
│
├── Telco-Customer-Churn-FromDB.csv            # Dataset
├── Telco_Churn.py                             # Main pipeline Python script
├── export_from_db.py                          # Database export script (fetches data directly from DB)
└── README.md                                  # Project documentation
