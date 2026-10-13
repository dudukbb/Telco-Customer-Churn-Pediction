##################################
# Telco Customer Churn Feature Engineering
##################################

# Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


# Gerekli Kütüphane ve Fonksiyonlar
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Çıktı klasörlerinin kontrolü ve oluşturulması (outputs, outputs/metrics, outputs/importances)
# Çıktı klasörlerinin kontrolü ve eksiksiz oluşturulması
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/importances", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)
os.makedirs("outputs/eda/categorical_plots", exist_ok=True)
os.makedirs("outputs/eda/numerical_plots", exist_ok=True)
os.makedirs("outputs/eda/target_analysis_numeric", exist_ok=True)
os.makedirs("outputs/eda/target_analysis_numeric_plots", exist_ok=True) # <--- BURAYA YAPIŞTIRACAKSIN
os.makedirs("outputs/eda/target_analysis_categoric", exist_ok=True)
os.makedirs("outputs/eda/target_analysis_categoric_plots", exist_ok=True)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].dtype

# Target değişken sayısal olmalı, ml modeli sayısal metrikle çalışır
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["Churn"].dtype


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=['number']).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)
df.head()

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# customerID : Eşsiz kimlik numarası (Unique Identifier).
# Modelin tahmin yeteneğine bir katkısı olmadığı ve her müşteri için farklı olduğu için
# ne kategorik (cat_cols) ne de numerik (num_cols) analizlere dahil edilmez;
# model kurulumu aşamasında veri setinden (X'ten) drop edilir (çıkarılır).
num_cols = [col for col in num_cols if col != "customerID"]
cat_cols = [col for col in cat_cols if col != "customerID"]

cat_cols
num_cols
cat_but_car


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, save_plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    plt.figure(figsize=(8, 6))
    sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.title(f'Count Plot - {col_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'outputs/eda/categorical_plots/{col_name}_countplot.png')
    plt.close()


for col in cat_cols:
    cat_summary(df, col, save_plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, save_plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    plt.figure(figsize=(8, 6))
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.title(f'Histogram - {numerical_col}')
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'outputs/eda/numerical_plots/{numerical_col}_histogram.png')
    plt.close()


for col in num_cols:
    num_summary(df, col, save_plot=True)


# Özel Karşılaştırma Histogramları ve Kayıtları
def save_contract_hist(dataframe, col_name, filename):
    plt.figure(figsize=(8, 6))
    dataframe[col_name].hist(bins=20)
    plt.xlabel(col_name)
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(f'outputs/eda/numerical_plots/{filename}.png')
    plt.close()


save_contract_hist(df[df["Contract"] == "Month-to-month"], "tenure", "tenure_Month_to_month")
save_contract_hist(df[df["Contract"] == "Two year"], "tenure", "tenure_Two_year")
save_contract_hist(df[df["Contract"] == "Month-to-month"], "MonthlyCharges", "MonthlyCharges_Month_to_month")
save_contract_hist(df[df["Contract"] == "Two year"], "MonthlyCharges", "MonthlyCharges_Two_year")

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    res_df = dataframe.groupby(target).agg({numerical_col: "mean"})
    print(res_df, end="\n\n\n")
    # CSV olarak kaydetme
    res_df.to_csv(f"outputs/eda/target_analysis_numeric/target_mean_num_{numerical_col}.csv")

    # Resim (Boxplot) olarak kaydetme
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target, y=numerical_col, data=dataframe)
    plt.title(f'Target Analysis (Boxplot) - {numerical_col}')
    plt.tight_layout()
    plt.savefig(f'outputs/eda/target_analysis_numeric_plots/target_mean_num_{numerical_col}.png')
    plt.close()


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    summary_df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                               "Count": dataframe[categorical_col].value_counts(),
                               "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)})
    print(summary_df, end="\n\n\n")
    # CSV olarak kaydetme
    summary_df.to_csv(f"outputs/eda/target_analysis_categoric/target_summary_cat_{categorical_col}.csv")

    # Resim (Barplot - TARGET_MEAN) olarak kaydetme
    plt.figure(figsize=(8, 6))
    sns.barplot(x=summary_df.index, y="TARGET_MEAN", data=summary_df, palette="viridis")
    plt.title(f'Target Mean by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel("Churn Rate (Target Mean)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/eda/target_analysis_categoric_plots/target_summary_cat_{categorical_col}.png')
    plt.close()


for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##################################
# KORELASYON
##################################

df[num_cols].corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.tight_layout()
plt.savefig('outputs/eda/numerical_plots/correlation_matrix.png')
plt.close()

df[num_cols].corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df.isnull().sum()
##################################
# BASE MODEL KURULUMU
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

base_results = []

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

    acc = round(cv_results['test_accuracy'].mean(), 4)
    auc = round(cv_results['test_roc_auc'].mean(), 4)
    recall = round(cv_results['test_recall'].mean(), 4)
    precision = round(cv_results['test_precision'].mean(), 4)
    f1 = round(cv_results['test_f1'].mean(), 4)

    base_results.append(
        {"Model": name, "Accuracy": acc, "AUC": auc, "Recall": recall, "Precision": precision, "F1": f1})

    print(f"########## {name} ##########")
    print(f"Accuracy: {acc}")
    print(f"Auc: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")

def save_dataframe_as_image(df, filename="outputs/metrics/base_models_summary.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1.5))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4c72b0')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f2f2f2' if key[0] % 2 == 0 else 'white')

    plt.title("Base Models Performance Summary", fontsize=14, weight='bold', pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# DataFrame'e dönüştürme, sıralama ve hem CSV hem PNG olarak kaydetme
df_base_results = pd.DataFrame(base_results).sort_values(by="AUC", ascending=False).reset_index(drop=True)

df_base_results.to_csv("outputs/metrics/base_models_summary.csv", index=False)
save_dataframe_as_image(df_base_results, filename="outputs/metrics/base_models_summary.png")


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

##################################
# ÖZELLİK ÇIKARIMI
##################################

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
            x["TechSupport"] != "Yes") else 0, axis=1)
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
                                       axis=1)

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()
df.shape

##################################
# ENCODING
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

##################################
# MODELLEME
##################################

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

engineered_results = []

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

    acc = round(cv_results['test_accuracy'].mean(), 4)
    auc = round(cv_results['test_roc_auc'].mean(), 4)
    recall = round(cv_results['test_recall'].mean(), 4)
    precision = round(cv_results['test_precision'].mean(), 4)
    f1 = round(cv_results['test_f1'].mean(), 4)

    engineered_results.append(
        {"Model": name, "Accuracy": acc, "AUC": auc, "Recall": recall, "Precision": precision, "F1": f1})

    print(f"########## {name} ##########")
    print(f"Accuracy: {acc}")
    print(f"Auc: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")

# Sonuçları DataFrame'e çevirip AUC'ye göre sıralıyoruz ve sadece PNG olarak kaydediyoruz
df_engineered_results = pd.DataFrame(engineered_results).sort_values(by="AUC", ascending=False).reset_index(drop=True)

save_dataframe_as_image(df_engineered_results, filename="outputs/metrics/engineered_models_summary.png")

print("Özellik mühendisliği sonrası model özetleri sadece PNG tablo olarak 'outputs/metrics/' klasörüne kaydedildi!")

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, 3],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# LightGBM
################################################

from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(random_state=17, verbose=-1)

# Parametreleri daha hızlı sonuç alacak şekilde daraltıyoruz
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300],
               "colsample_bytree": [0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=3, n_jobs=-1, verbose=2).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17, verbose=-1).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print("Accuracy:", cv_results['test_accuracy'].mean())
print("F1-Score:", cv_results['test_f1'].mean())
print("ROC-AUC:", cv_results['test_roc_auc'].mean())

################################################
# CatBoost
################################################

from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# Feature Importance ve Otomatik Kayıt
################################################

def plot_importance(model, features, model_name, num=20, save=True):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    if save:
        plt.savefig(f'outputs/importances/{model_name}_importance.png')
    plt.close()


plot_importance(rf_final, X, "RandomForest")
plot_importance(xgboost_final, X, "XGBoost")
plot_importance(lgbm_final, X, "LightGBM")
plot_importance(catboost_final, X, "CatBoost")

print("Tüm metrik özetleri 'outputs/metrics/' ve grafikler 'outputs/importances/' klasörüne başarıyla kaydedildi!")

