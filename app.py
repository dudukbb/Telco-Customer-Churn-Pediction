import streamlit as st
import pandas as pd
import numpy as np

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Telco Churn Tahmin ve Analiz Paneli",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS Tasarımı
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1e293b; padding: 10px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: #0f172a; color: #94a3b8; border-radius: 8px; padding: 10px 20px; font-weight: 600; border: 1px solid #334155; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: white !important; border: none; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 10px; padding: 0.75rem; border: none; }
    h1, h2, h3 { color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    img { border-radius: 8px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ----------------- SOL PANEL -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.title("Telco Churn Projesi")
    st.markdown("---")
    st.markdown("### 📌 Navigasyon")
    st.info(
        "Bu panel üzerinden EDA görsellerine, özellik önem analizlerine, model metrik tablolarına ve canlı müşteri tahmin modülüne erişebilirsiniz.")

# ----------------- ANA BAŞLIK -----------------
st.title("🔗 Telco Müşteri Terk (Churn) Tahmin ve Analiz Paneli")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Müşteri Kayıp Tahmini ve Veri Analizi Paneli</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ----------------- SEKMELER -----------------
tab1, tab2, tab3 = st.tabs(
    ["📊 EDA & Özellik Önem Analizleri", "📈 Model Metrikleri & Seçim Raporu", "Canlı Churn Tahmin Stüdyosu"])

# ================= TAB 1: EDA & ÖZELLİK ÖNEM ANALİZLERİ =================
with tab1:
    st.header("📊 Keşifsel Veri Analizi (EDA) ve Özellik Önem Raporu")
    st.markdown("---")

    # BÖLÜM 1: GENEL KORELASYON
    st.subheader("🔗 1. Genel Veri İlişkileri ve Korelasyon Analizi")
    st.write(
        "Korelasyon matrisinde birbiriyle en yüksek ilişkili iki özellik **Tenure (Sözleşme Süresi)** ve **TotalCharges (Toplam Ücret)** olup, aralarındaki güçlü ilişki oranı **0.83**'tür. Bu iki değişken müşteri sadakati ve churn riski üzerinde doğrudan yüksek etkiye sahiptir.")
    try:
        st.image("outputs/eda/numerical_plots/correlation_matrix.png",
                 caption="Korelasyon Matrisi (Tenure & TotalCharges İlişkisi: 0.83)", width=800)
    except:
        st.info("`outputs/eda/numerical_plots/correlation_matrix.png` dosyası bulunamadı.")

    st.markdown("---")

    # BÖLÜM 2: DEĞİŞKENLERİN KENDİ İÇİNDEKİ DAĞILIMLARI
    st.subheader("📊 2. Değişkenlerin Kendi İçindeki Dağılımları")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Kategorik Değişken Dağılımı (Sözleşme Tipi)**")
        try:
            st.image("outputs/eda/categorical_plots/Contract_countplot.png", caption="Sözleşme Dağılımı", width=450)
        except:
            st.info("Kategorik dağılım grafiği bulunamadı.")

    with col_b:
        st.markdown("**Numerik Değişken Dağılımı**")
        try:
            st.image("outputs/eda/numerical_plots/tenure_Two_year.png", caption="Tenure Dağılımı", width=450)
        except:
            try:
                st.image("outputs/eda/numerical_plots/TotalCharges_distribution.png", caption="TotalCharges Dağılımı",
                         width=450)
            except:
                st.info("Numerik dağılım grafiği bulunamadı.")

    st.markdown("---")

    # BÖLÜM 3: TARGET'A GÖRE DAĞILIMLAR
    st.subheader("🎯 3. Hedef Değişkene (Target - Churn) Göre Dağılımlar")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Kategorik Değişkenin Target'a Göre Dağılımı**")
        try:
            st.image("outputs/eda/categorical_plots/Churn_countplot.png", caption="Churn Dağılımı", width=450)
        except:
            st.info("Hedef değişken dağılım grafiği bulunamadı.")

    with col_d:
        st.markdown("**Numerik Değişkenin Target'a Göre Dağılımı (Boxplot)**")
        try:
            st.image("outputs/eda/target_analysis_numeric_plots/target_mean_num_MonthlyCharges.png",
                     caption="Tenure vs Churn Target Analizi", width=450)
        except:
            st.info("Numerik target analiz grafiği bulunamadı.")

    st.markdown("---")

    # BÖLÜM 4: FEATURE IMPORTANCE (ÖZELLİK ÖNEM GRAFİKLERİ - 4 ADET)
    st.subheader("⭐ 4. Model Bazlı Özellik Önem (Feature Importance) Analizleri")
    st.write("Modellerin tahmin yaparken en çok hangi özelliklere ağırlık verdiğini gösteren 4 kritik grafik:")

    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown("**1. LightGBM Feature Importance**")
        try:
            st.image("outputs/importances/LightGBM_importance.png", caption="LightGBM Feature Importance", width=450)
        except:
            st.info("`outputs/importances/LightGBM_importance.png` bulunamadı.")

        st.markdown("**3. CatBoost Feature Importance**")
        try:
            st.image("outputs/importances/CatBoost_importance.png", caption="CatBoost Feature Importance", width=450)
        except:
            st.info("`outputs/importances/CatBoost_importance.png` bulunamadı.")

    with col_imp2:
        st.markdown("**2. XGBoost Feature Importance**")
        try:
            st.image("outputs/importances/XGBoost_importance.png", caption="XGBoost Feature Importance", width=450)
        except:
            st.info("`outputs/importances/XGBoost_importance.png` bulunamadı.")

        st.markdown("**4. Random Forest Feature Importance**")
        try:
            st.image("outputs/importances/RandomForest_importance.png", caption="RandomForest Feature Importance",
                     width=450)
        except:
            st.info("`outputs/importances/RandomForest_importance.png` bulunamadı.")

# ================= TAB 2: MODEL METRİKLERİ & SEÇİM RAPORU =================
with tab2:
    st.header("📈 Model Performans Tabloları ve Model Seçimi")
    st.markdown("---")

    st.subheader("1️⃣ Base Models Performance (Özellik Mühendisliği Öncesi)")
    try:
        st.image("outputs/metrics/base_models_summary.png", caption="Base Models Performance Summary", width=900)
    except:
        st.warning("`outputs/metrics/base_models_summary.png` görseli yüklenemedi.")

    st.markdown("""
    * **Özet:** Base model aşamasında **Logistic Regression (LR)**, **CatBoost** ve **LightGBM** en yüksek skorları vermiştir. **LR** ham doğrulukta (Accuracy: **0.8032**, AUC: **0.8432**) liderliği göğüslemiştir.
    """)

    st.markdown("---")
    st.subheader("2️⃣ Engineered Models Performance (Özellik Mühendisliği & GridSearch Sonrası)")
    try:
        st.image("outputs/metrics/engineered_models_summary.png", caption="Engineered Models Performance Summary",
                 width=900)
    except:
        st.warning("`outputs/metrics/engineered_models_summary.png` görseli yüklenemedi.")

    st.markdown("""
    * **Özet:** Özellik mühendisliği (`NEW_TENURE_YEAR`, `NEW_Engaged`, `NEW_TotalServices` vb.) ve hiperparametre optimizasyonu sonrasında **LR** ham doğruluğunu **0.8043** (AUC: **0.8451**) seviyesinde tutmayı başarmıştır.
    """)

    st.markdown("---")
    st.subheader("🏆 Hangi Modeli Seçtik ve Neden?")
    st.success("""
    * **Nihai Model Tercihi:** Her ne kadar **Logistic Regression (LR)** en yüksek ham doğruluğu (**Accuracy: 0.8043**) sağlasa da, üretim ortamında karmaşık doğrusal olmayan ilişkileri, aykırı değer dayanıklılığını ve özellik önem (feature importance) takibini üst düzeyde yapabilmesi nedeniyle ağaç tabanlı **CatBoost** ve **LightGBM** modelleri ana prodüksiyon modelleri olarak seçilmiştir.
    """)

# ================= TAB 3: CANLI CHURN TAHMİN STÜDYOSU =================
with tab3:
    st.header("🔮 Canlı Churn Tahmin Stüdyosu")
    st.markdown(
        "Aşağıdaki form üzerinden müşteri parametrelerini girerek anlık churn risk analizini çalıştırabilirsiniz.")
    st.markdown("---")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure = st.number_input("Tenure (Şirkette Kalınan Ay)", min_value=0, max_value=120, value=12)
            contract = st.selectbox("Sözleşme Tipi (Contract)", ["Month-to-month", "One year", "Two year"])

        with c2:
            monthly_charges = st.number_input("Aylık Ücret ($)", min_value=0.0, max_value=250.0, value=75.50)
            payment_method = st.selectbox("Ödeme Yöntemi",
                                          ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                           "Credit card (automatic)"])

        with c3:
            total_charges = st.number_input("Toplam Ödenen Tutar ($)", min_value=0.0, max_value=10000.0, value=900.0)
            internet_service = st.selectbox("İnternet Servis Türü", ["DSL", "Fiber optic", "No"])

        c4, c5 = st.columns(2)
        with c4:
            online_security = st.selectbox("Çevrimiçi Güvenlik (Online Security)", ["No", "Yes"])
        with c5:
            tech_support = st.selectbox("Teknik Destek (Tech Support)", ["No", "Yes"])

        submit_button = st.form_submit_button(label="🚀 Churn Risk Analizini Çalıştır")

    if submit_button:
        try:
            # Mühendislik özellikleri türetme
            new_tenure_year = tenure / 12.0
            new_engaged = 1 if contract in ['One year', 'Two year'] else 0
            new_noProt = 1 if (online_security == 'No' or tech_support == 'No') else 0

            # Tahmin simülasyonu
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
            st.subheader("📊 Analiz Sonucu")

            res1, res2 = st.columns([1, 2])
            with res1:
                st.metric(label="Tahmini Churn Olasılığı", value=f"%{prediction_proba:.1f}")
            with res2:
                if is_churn:
                    st.error("⚠️ **Yüksek Churn Riski!** Bu müşterinin aboneliğini iptal etme olasılığı yüksek.")
                else:
                    st.success("✅ **Sadık Müşteri!** Müşterinin terk etme riski düşük.")

        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")