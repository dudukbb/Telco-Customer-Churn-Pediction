# Telco-Customer-Churn-Pediction

# Telco Customer Churn Prediction & Database Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PostgreSQL-Database-orange.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20LightGBM%20%7C%20CatBoost-green.svg" alt="ML Libraries">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">
</p>

## 📌 Proje Hakkında
Bu proje, telekomünikasyon sektöründe müşteri kaybını (**Churn**) önceden tahmin etmek amacıyla geliştirilmiş uçtan uca bir **Veri Bilimi ve Veri Mühendisliği** çalışmasıdır. Projede yalnızca makine öğrenmesi modelleri eğitilmekle kalmamış; verilerin güvenli bir şekilde ilişkisel veritabanında saklanması, SQL ile yönetilmesi ve Python betikleriyle otomatize edilmesi sağlanmıştır.

---

## 🗄️ Veritabanı ve Mimari Yapısı (Data Engineering)
Proje, gerçek dünya senaryolarına uygun olarak bir veritabanı katmanı ile desteklenmiştir:
1. **PostgreSQL Entegrasyonu (`telco_churn_db`)**: Ham veri seti, yerel PostgreSQL veritabanında oluşturulan `telco_customers` tablosuna aktarılmıştır.
2. **Otomatik Tablo Kurulumu (`create_table.py`)**: Tablo şemasını otomatik oluşturan ve 7,043 satırlık veriyi veritabanına `INSERT` eden Python betiği.
3. **Veri Dışa Aktarımı (`export_from_db.py`)**: Modelleme aşamasından önce verilerin SQL sorgusuyla veritabanından çekilip Pandas DataFrame'e aktarılmasını sağlayan katman.

---

## 📂 Proje Dosya Yapısı
```text
Telco_Customer_Churn_Prediction/
│
├── .venv/                      # Python Sanal Ortamı
├── outputs/                    # Çıktılar ve Model Önem Grafikleri
├── create_table.py             # PostgreSQL tablo oluşturma ve veri yükleme betiği
├── export_from_db.py           # Veritabanından veriyi çekme betiği
├── telco_churn.py              # EDA, Feature Engineering ve Model Eğitimi ana betiği
├── Telco-Customer-Churn.csv    # Ham veri seti
├── Telco-Customer-Churn-FromDB.csv # Veritabanından dışarı aktarılan veri seti
└── README.md                   # Proje Dokümantasyonu
