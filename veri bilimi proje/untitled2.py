# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:59:08 2025

@author:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 1. Veriyi oku
df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

# 2. Tarih sütununu işle
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df = df.drop("Date", axis=1)

# 3. Kategorik dönüşümler ve sütun adları
df["Seasons"] = df["Seasons"].astype("category")
df["Holiday"] = df["Holiday"].astype("category")
df["Functioning Day"] = df["Functioning Day"].astype("category")
df = df.rename(columns={"Rented Bike Count": "BikeCount"})

# 4A. Aylara göre ortalama kiralama
plt.figure(figsize=(10, 6))
sns.barplot(x="Month", y="BikeCount", data=df, ci=None, estimator=np.mean)
plt.title("Aylara Göre Ortalama Bisiklet Kiralama", fontsize=14)
plt.xlabel("Ay", fontsize=12)
plt.ylabel("Ortalama Kiralama", fontsize=12)
plt.xticks(ticks=range(0, 12), labels=["Ocak", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"])
plt.tight_layout()
plt.savefig("aylik_kiralama.png")
plt.close()

# 4B. Sıcaklığa göre kiralama
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Temperature(°C)", y="BikeCount", data=df, alpha=0.4)
plt.title("Sıcaklık ve Bisiklet Kiralama İlişkisi", fontsize=14)
plt.xlabel("Sıcaklık (°C)", fontsize=12)
plt.ylabel("Bisiklet Kiralama", fontsize=12)
plt.tight_layout()
plt.savefig("sicaklik_kiralama.png")
plt.close()

# 4C. Mevsimlere göre kiralama
plt.figure(figsize=(10, 6))
sns.boxplot(x="Seasons", y="BikeCount", data=df)
plt.title("Mevsimlere Göre Bisiklet Kiralama Dağılımı", fontsize=14)
plt.xlabel("Mevsim", fontsize=12)
plt.ylabel("Bisiklet Kiralama", fontsize=12)
plt.tight_layout()
plt.savefig("mevsim_kiralama.png")
plt.close()

# 4D. Haftanın günlerine göre kiralama
plt.figure(figsize=(10, 6))
sns.boxplot(x="DayOfWeek", y="BikeCount", data=df)
plt.title("Haftanın Günlerine Göre Bisiklet Kiralama", fontsize=14)
plt.xlabel("Gün (0 = Pazartesi)", fontsize=12)
plt.ylabel("Kiralama", fontsize=12)
plt.tight_layout()
plt.savefig("gunluk_kiralama.png")
plt.close()

# 4E. Korelasyon matrisi (üst üste binmeleri önlemek için geniş font ve boyut)
plt.figure(figsize=(14, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 9})
plt.title("Sayısal Değişkenler Korelasyon Matrisi", fontsize=16)
plt.tight_layout()
plt.savefig("korelasyon_matrisi.png")
plt.close()

# 5. Tatil günleri hipotez testi
holiday_group = df[df["Holiday"] == "Holiday"]["BikeCount"]
no_holiday_group = df[df["Holiday"] == "No Holiday"]["BikeCount"]

t_stat, p_value = ttest_ind(holiday_group, no_holiday_group, equal_var=False)

print("\nHipotez Testi Sonuçları (Tatil vs. Tatil Değil):")
print(f"T-istatistiği: {t_stat:.3f}")
print(f"P-değeri: {p_value:.5f}")

# 6. Tatil günleri karşılaştırma grafiği
plt.figure(figsize=(8, 6))
sns.barplot(x="Holiday", y="BikeCount", data=df, ci="sd", palette="Set2")
plt.title("Tatil Günlerinde Kiralama Ortalaması", fontsize=14)
plt.xlabel("Gün Türü", fontsize=12)
plt.ylabel("Bisiklet Kiralama", fontsize=12)
plt.text(0.5, df["BikeCount"].max() * 0.9, f"p-değeri = {p_value:.4f}", ha="center", fontsize=12, color="red")
plt.tight_layout()
plt.savefig("hipotez_tatil_karsilastirmasi.png")
plt.close()

# 7. Ortalama ve yüzde değişim
mean_holiday = holiday_group.mean()
mean_no_holiday = no_holiday_group.mean()
percent_change = ((mean_holiday - mean_no_holiday) / mean_no_holiday) * 100

print(f"Tatil günlerindeki ortalama kiralama: {mean_holiday:.2f}")
print(f"Tatil olmayan günlerdeki ortalama kiralama: {mean_no_holiday:.2f}")
print(f"Yüzde değişim: %{percent_change:.2f}")

print("\nTüm grafikler başarıyla oluşturuldu ve kaydedildi.")