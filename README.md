# 📉 Telco Customer Churn Prediction 

End-to-end churn prediction pipeline featuring EDA, feature engineering, and hyperparameter tuning with top ML models."

---

## 🔗 Project Overview & My Workflow
I designed this project to cover the entire machine learning lifecycle step by step:

1. **Exploratory Data Analysis (EDA):** I inspected the data structure, analyzed categorical and numerical distributions, and visualized their relationships against the target variable (`Churn`).
2. **Missing & Outlier Value Analysis:** I handled missing values through proper imputation and suppressed outliers based on custom statistical thresholds.
3. **Feature Engineering:** I engineered domain-specific features to capture deep behavioral patterns and enhance model predictive power.
4. **Modeling & Hyperparameter Tuning:** I benchmarked baseline models and optimized top-tier algorithms (`RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`) using `GridSearchCV` and k-fold cross-validation.
5. **Automated Output Management:** I implemented automated storage so that all performance metric tables (`.png`) and feature importance charts (`.png`) are saved directly into structured directories.

---

## 🔗 Feature Engineering
To capture customer loyalty and behavioral trends, I derived the following features:
* **`NEW_TENURE_YEAR`**: Grouped customer tenure into yearly bins (0-1 year up to 5-6 years).
* **`NEW_Engaged`**: Binary flag indicating if the customer holds a long-term contract (One or Two years).
* **`NEW_noProt`**: Indicator for customers lacking essential online security, backup, or tech support.
* **`NEW_TotalServices`**: Total count of active services subscribed by the customer.
* **`NEW_FLAG_ANY_STREAMING`**: Flag indicating usage of either TV or movie streaming services.
* **`NEW_FLAG_AutoPayment`**: Indicator for automated payment methods (Credit card or Bank transfer).
* **`NEW_AVG_Charges` & `NEW_Increase`**: Financial metrics tracking charge behaviors over tenure.
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
