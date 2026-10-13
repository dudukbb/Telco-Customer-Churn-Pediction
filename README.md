# 🚀 Telco Customer Churn Prediction & Feature Engineering Pipeline

Bu proje, telekomünikasyon sektöründeki müşteri verilerini kullanarak hangi müşterilerin şirketi terk edeceğini (`churn`) tahmin eden, uçtan uca kapsamlı bir Makine Öğrenmesi ve Veri Bilimi pipeline çalışmasıdır.

---

## 📌 Proje Özeti ve Kapsamı
Proje; ham verinin incelenmesinden (EDA), ileri düzey özellik mühendisliğine, temel model kıyaslamalarından, hiperparametre optimizasyonuna ve otomatik çıktı kaydetmeye kadar tüm adımları kapsamaktadır. 

### Temel Adımlar:
1. **Keşifsel Veri Analizi (EDA):** Veri setinin genel yapısı incelenmiş, kategorik ve numerik değişkenler analiz edilmiş, target (hedef) değişkenine göre kırılımlar incelenerek görselleştirilmiştir.
2. **Eksik ve Aykırı Değer Analizi:** Eksik değerler uygun yöntemlerle doldurulmuş, eşik değerleri (thresholds) belirlenerek aykırı gözlemler baskılanmıştır.
3. **Özellik Mühendisliği (Feature Engineering):** Müşteri davranışlarını modelin daha iyi anlaması için domain bilgisine dayalı yeni değişkenler türetilmiştir.
4. **Modelleme & Hiperparametre Optimizasyonu:** `RandomForest`, `XGBoost`, `LightGBM` ve `CatBoost` gibi güçlü algoritmalar `GridSearchCV` ve çapraz doğrulama (`cross_validate`) ile optimize edilmiştir.
5. **Otomatik Çıktı Yönetimi:** Elde edilen tüm performans metrik tabloları (CSV/PNG) ve değişken önem (`feature importance`) grafikleri otomatik olarak `outputs/` klasörüne kaydedilmektedir.

---

## ⚙️ Türetilen Yeni Değişkenler (Feature Engineering)
Projede müşteri sadakatini ve davranış kalıplarını yakalamak için üretilen türetilmiş öznitelikler:
* **`NEW_TENURE_YEAR`**: Müşterinin şirketteki kalış süresinin (tenure) yıllara göre kategorize edilmiş hali (0-1 yıl, 1-2 yıl ... 5-6 yıl).
* **`NEW_Engaged`**: Müşterinin uzun süreli sözleşmeye (1 veya 2 yıllık) sahip olup olmadığını gösteren ikili (binary) bayrak.
* **`NEW_noProt`**: Çevrimiçi güvenlik, yedekleme veya teknik destek hizmetlerinden en az birine sahip olmayan müşterileri işaret eden öznitelik.
* **`NEW_TotalServices`**: Müşterinin aktif olarak kullandığı toplam servis sayısı.
* **`NEW_FLAG_ANY_STREAMING`**: TV veya film akış hizmetlerinden herhangi birini kullanıp kullanmadığının kontrolü.
* **`NEW_FLAG_AutoPayment`**: Otomatik ödeme yöntemi (Kredi kartı veya Banka havalesi) kullanıp kullanmadığı.
* **`NEW_AVG_Charges` & `NEW_Increase`**: Müşterinin aylık ve toplam ödemeleri üzerinden finansal yük davranışlarını takip eden oranlar.
* **`NEW_AVG_Service_Fee`**: Aldığı hizmet başına ödediği ortalama ücret.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Python** 
* **Veri Analizi:** `Pandas`, `NumPy`
* **Görselleştirme:** `Matplotlib`, `Seaborn`
* **Makine Öğrenmesi & Modelleme:** 
  * `Scikit-Learn` (`GridSearchCV`, `cross_validate`, `LabelEncoder`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`)
  * `XGBoost`
  * `LightGBM`
  * `CatBoost`

---

## 📂 Proje Dizin Yapısı
```text
Telco_Customer_Churn_Prediction/
│
├── outputs/
│   ├── eda/                                   # EDA dağılım ve hedef analizi grafikleri
│   │   ├── categorical_plots/
│   │   ├── numerical_plots/
│   │   ├── target_analysis_categoric/
│   │   ├── target_analysis_categoric_plots/
│   │   ├── target_analysis_numeric/
│   │   └── target_analysis_numeric_plots/
│   ├── importances/                           # Model özellik önem (feature importance) görselleri
│   │   ├── RandomForest_importance.png
│   │   ├── XGBoost_importance.png
│   │   ├── LightGBM_importance.png
│   │   └── CatBoost_importance.png
│   └── metrics/                               # Model performans özet tabloları (CSV ve PNG)
│       ├── base_models_summary.csv / .png
│       └── engineered_models_summary.png
│
├── Telco-Customer-Churn.csv                   # Veri seti
├── Telco_Churn.py                             # Ana Python betiği (Pipeline)
└── README.md                                  # Proje dokümantasyonu
