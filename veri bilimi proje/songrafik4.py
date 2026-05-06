# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:42:24 2025

@author:
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 1. Veri setini yükle
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')

# 2. Tarih verisini işle ve sütun adlarını düzenle
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})  # Sütun adını değiştiriyoruz

# 3. Kategorik dönüşümler
df['Seasons'] = df['Seasons'].astype('category')
df['Holiday'] = df['Holiday'].astype('category')
df['Functioning Day'] = df['Functioning Day'].astype('category')

# 4. Hipotez testi (Tatil vs Tatil Değil)
holiday_group = df[df['Holiday'] == 'Holiday']['BikeCount']
no_holiday_group = df[df['Holiday'] == 'No Holiday']['BikeCount']

t_stat, p_value = ttest_ind(holiday_group, no_holiday_group, equal_var=False)
mean_holiday = holiday_group.mean()
mean_no_holiday = no_holiday_group.mean()
percent_change = ((mean_holiday - mean_no_holiday) / mean_no_holiday) * 100

print("\nHipotez Testi Sonuçları (Tatil vs. Tatil Değil):")
print(f"T-istatistiği: {t_stat:.3f}")
print(f"P-değeri: {p_value:.5f}")
print(f"Tatil günlerindeki ortalama kiralama: {mean_holiday:.2f}")
print(f"Tatil olmayan günlerdeki ortalama kiralama: {mean_no_holiday:.2f}")
print(f"Yüzde değişim: %{percent_change:.2f}")

# 5. Haftanın günlerine göre ortalama kiralama (DÜZELTİLMİŞ KISIM)
plt.figure(figsize=(8, 5))
sns.barplot(x='DayOfWeek', y='BikeCount', data=df, estimator='mean', ci=None)  # BikeCount kullanıyoruz
plt.xticks(ticks=range(7), labels=['Pzt', 'Salı', 'Çar', 'Per', 'Cuma', 'Cmt', 'Paz'])
plt.title('Haftanın Günlerine Göre Ortalama Kiralama')
plt.ylabel('Ortalama Kiralama Sayısı')
plt.xlabel('Gün')
plt.tight_layout()
plt.savefig("hafta_gunlerine_gore_kiralama.png")
plt.close()

# 6. Diğer grafikler (hepsinde BikeCount kullanıyoruz)
plt.figure(figsize=(10, 5))
sns.boxplot(x='Hour', y='BikeCount', data=df)
plt.title('Saatlik Kiralama Dağılımı')
plt.xlabel('Saat')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("saatlik_kiralama_dagilimi.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Temperature(°C)', y='BikeCount', data=df, alpha=0.5)
plt.title('Sıcaklık ile Kiralama İlişkisi')
plt.xlabel('Sıcaklık (°C)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("sicaklik_kiralama_iliski.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Humidity(%)', y='BikeCount', data=df, alpha=0.5)
plt.title('Nem ile Kiralama İlişkisi')
plt.xlabel('Nem (%)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("nem_kiralama_iliski.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Seasons', y='BikeCount', data=df, estimator='mean', ci=None)
plt.title('Mevsimlere Göre Ortalama Kiralama')
plt.ylabel('Ortalama Kiralama')
plt.xlabel('Mevsim')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mevsimlere_gore_kiralama.png")
plt.close()

print("Tüm grafikler başarıyla oluşturuldu ve kaydedildi.")