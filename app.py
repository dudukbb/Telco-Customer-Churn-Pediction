import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Prediction & Analysis Panel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Soft CSS Design (Light & Soft Theme)
st.markdown("""
    <style>
    /* General Background and Text Color - Soft Gray */
    .stApp { background-color: #f8fafc !important; color: #1e293b !important; }
    .main { background-color: #f8fafc !important; color: #1e293b !important; }

    /* Sidebar Design - Soft / Light Color */
    section[data-testid="stSidebar"] { 
        background-color: #f1f5f9 !important; 
        border-right: 1px solid #e2e8f0; 
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p {
        color: #334155 !important;
    }

    /* Tabs Design - Soft Look */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background-color: #e2e8f0 !important; 
        padding: 8px; 
        border-radius: 12px; 
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff !important; 
        color: #64748b !important; 
        border-radius: 8px; 
        padding: 10px 20px; 
        font-weight: 600; 
        border: 1px solid #cbd5e1; 
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; 
        color: white !important; 
        border: none !important; 
    }

    /* Button Design */
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
        color: white; 
        font-weight: 700; 
        border-radius: 10px; 
        padding: 0.75rem; 
        border: none; 
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }

    /* Headers and Text */
    h1, h2, h3, h4, h5, h6, p, span { 
        color: #0f172a !important; 
        font-family: 'Segoe UI', sans-serif; 
    }

    /* Image Frames */
    img { border-radius: 8px; margin-bottom: 15px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.title("Telco Churn Project")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.info(
        "Use this panel to explore EDA visualizations, feature importance reports, model metric tables, and the live customer prediction module.")

# ----------------- MAIN TITLE (İkonsuz) -----------------
st.title("Telco Customer Churn Prediction")
st.markdown(
    "<p style='color: #64748b !important; font-size: 1.1rem;'>Customer Loss Prediction and Data Analysis Dashboard</p>",
    unsafe_allow_html=True)
st.markdown("---")

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(
    ["📊 EDA & Feature Importance", "📈 Model Metrics & Selection", "Live Churn Prediction Studio"])

# ================= TAB 1: EDA & FEATURE IMPORTANCE =================
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA) & Feature Importance")
    st.markdown("---")

    # SECTION 1: CORRELATION
    st.subheader("1. General Data Relations and Correlation Analysis")
    st.write(
        "The highest correlated features in the matrix are **Tenure** and **TotalCharges**, with a strong correlation coefficient of **0.83**. These two variables have a direct and significant impact on customer loyalty and churn risk.")
    try:
        st.image("outputs/eda/numerical_plots/correlation_matrix.png",
                 caption="Correlation Matrix (Tenure & TotalCharges Relation: 0.83)", width=800)
    except:
        st.info("`outputs/eda/numerical_plots/correlation_matrix.png` file not found.")

    st.markdown("---")

    # SECTION 2: FEATURE DISTRIBUTIONS
    st.subheader("2. Individual Feature Distributions")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Categorical Distribution (Contract Type)**")
        try:
            st.image("outputs/eda/categorical_plots/Contract_countplot.png", caption="Contract Distribution", width=450)
        except:
            st.info("Categorical distribution plot not found.")

    with col_b:
        st.markdown("**Numerical Distribution**")
        try:
            st.image("outputs/eda/numerical_plots/tenure_Two_year.png", caption="Tenure Distribution", width=450)
        except:
            try:
                st.image("outputs/eda/numerical_plots/TotalCharges_distribution.png",
                         caption="TotalCharges Distribution",
                         width=450)
            except:
                st.info("Numerical distribution plot not found.")

    st.markdown("---")

    # SECTION 3: DISTRIBUTIONS BY TARGET
    st.subheader("3. Distributions by Target Variable (Churn)")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Categorical Feature by Target**")
        try:
            st.image("outputs/eda/categorical_plots/Churn_countplot.png", caption="Churn Distribution", width=450)
        except:
            st.info("Target distribution plot not found.")

    with col_d:
        st.markdown("**Numerical Feature by Target (Boxplot)**")
        try:
            st.image("outputs/eda/target_analysis_numeric_plots/target_mean_num_MonthlyCharges.png",
                     caption="Tenure vs Churn Target Analysis", width=450)
        except:
            st.info("Numerical target analysis plot not found.")

    st.markdown("---")

    # SECTION 4: FEATURE IMPORTANCE
    st.subheader("4. Model-Based Feature Importance Analyses")
    st.write("Four critical charts showing which features models rely on the most:")

    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown("**1. LightGBM Feature Importance**")
        try:
            st.image("outputs/importances/LightGBM_importance.png", caption="LightGBM Feature Importance", width=450)
        except:
            st.info("`outputs/importances/LightGBM_importance.png` not found.")

        st.markdown("**3. CatBoost Feature Importance**")
        try:
            st.image("outputs/importances/CatBoost_importance.png", caption="CatBoost Feature Importance", width=450)
        except:
            st.info("`outputs/importances/CatBoost_importance.png` not found.")

    with col_imp2:
        st.markdown("**2. XGBoost Feature Importance**")
        try:
            st.image("outputs/importances/XGBoost_importance.png", caption="XGBoost Feature Importance", width=450)
        except:
            st.info("`outputs/importances/XGBoost_importance.png` not found.")

        st.markdown("**4. Random Forest Feature Importance**")
        try:
            st.image("outputs/importances/RandomForest_importance.png", caption="RandomForest Feature Importance",
                     width=450)
        except:
            st.info("`outputs/importances/RandomForest_importance.png` not found.")

# ================= TAB 2: MODEL METRICS & SELECTION =================
with tab2:
    st.header("📈 Model Performance Tables & Selection Report")
    st.markdown("---")

    st.subheader("1️⃣ Base Models Performance (Before Feature Engineering)")
    try:
        st.image("outputs/metrics/base_models_summary.png", caption="Base Models Performance Summary", width=900)
    except:
        st.warning("`outputs/metrics/base_models_summary.png` image could not be loaded.")

    st.markdown("""
    * **Summary:** At the base model stage, **Logistic Regression (LR)**, **CatBoost**, and **LightGBM** achieved the highest scores. **LR** took the lead in raw accuracy (Accuracy: **0.8032**, AUC: **0.8432**).
    """)

    st.markdown("---")
    st.subheader("2️⃣ Engineered Models Performance (After Feature Engineering & GridSearch)")
    try:
        st.image("outputs/metrics/engineered_models_summary.png", caption="Engineered Models Performance Summary",
                 width=900)
    except:
        st.warning("`outputs/metrics/engineered_models_summary.png` image could not be loaded.")

    st.markdown("""
    * **Summary:** Following feature engineering (`NEW_TENURE_YEAR`, `NEW_Engaged`, `NEW_TotalServices`, etc.) and hyperparameter optimization, **LR** maintained its raw accuracy at **0.8043** (AUC: **0.8451**).
    """)

    st.markdown("---")
    st.subheader("🏆 Which Model Did We Choose and Why?")
    st.success("""
    * **Final Model Choice:** Although **Logistic Regression (LR)** provided the highest raw accuracy (**Accuracy: 0.8043**), tree-based models **CatBoost** and **LightGBM** were chosen as primary production models due to their superior capability in handling complex non-linear relationships, outlier robustness, and feature importance tracking in production environments.
    """)

# ================= TAB 3: LIVE CHURN PREDICTION STUDIO =================
with tab3:
    st.header("🔮 Live Churn Prediction Studio")
    st.markdown(
        "Run an instant churn risk analysis by entering customer parameters through the form below.")
    st.markdown("---")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure = st.number_input("Tenure (Months with Company)", min_value=0, max_value=120, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

        with c2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=250.0, value=75.50)
            payment_method = st.selectbox("Payment Method",
                                          ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                           "Credit card (automatic)"])

        with c3:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=900.0)
            internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])

        c4, c5 = st.columns(2)
        with c4:
            online_security = st.selectbox("Online Security", ["No", "Yes"])
        with c5:
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])

        submit_button = st.form_submit_button(label="🚀 Run Churn Risk Analysis")

    if submit_button:
        try:
            # Feature engineering simulation
            new_tenure_year = tenure / 12.0
            new_engaged = 1 if contract in ['One year', 'Two year'] else 0
            new_noProt = 1 if (online_security == 'No' or tech_support == 'No') else 0

            # Prediction simulation
            risk_score = 15.0
            if contract == 'Month-to-month':
                risk_score += 40.0
            if monthly_charges > 75:
                risk_score += 20.0
            if tenure < 10:
                risk_score += 25.0

            prediction_proba = min(max(risk_score, 3.5), 96.8)
            is_churn = prediction_proba > 50.0

            st.markdown("---")
            st.subheader("📊 Analysis Result")

            res1, res2 = st.columns([1, 2])
            with res1:
                st.metric(label="Estimated Churn Probability", value=f"%{prediction_proba:.1f}")
            with res2:
                if is_churn:
                    st.error(
                        "⚠️ **High Churn Risk!** This customer has a high probability of canceling their subscription.")
                else:
                    st.success("✅ **Loyal Customer!** The customer has a low risk of churning.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
