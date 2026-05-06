# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:46:48 2025

@author:
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Veri setini yükle
df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')

# 2. Tarihleri ve sayısal sütunları işle
df.rename(columns={'Rented Bike Count': 'BikeCount'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Solar Radiation (MJ/m2)'] = pd.to_numeric(df['Solar Radiation (MJ/m2)'], errors='coerce')
df['Rainfall(mm)'] = pd.to_numeric(df['Rainfall(mm)'], errors='coerce')
df['Snowfall (cm)'] = pd.to_numeric(df['Snowfall (cm)'], errors='coerce')

# 3. Güneşli günleri seç (güneş var, yağmur yok, kar yok)
sunny_days = df[(df['Solar Radiation (MJ/m2)'] > 0.5) &
                (df['Rainfall(mm)'] == 0) &
                (df['Snowfall (cm)'] == 0)]

# 4. Güneşli olmayan günleri al
not_sunny_days = df.drop(sunny_days.index)

# 5. Ortalama kullanıcı sayısını hesapla
avg_sunny = sunny_days['BikeCount'].mean()
avg_not_sunny = not_sunny_days['BikeCount'].mean()
increase_ratio = ((avg_sunny - avg_not_sunny) / avg_not_sunny) * 100

print(f"Güneşli günlerde ortalama bisiklet sayısı: {avg_sunny:.2f}")
print(f"Güneşsiz günlerde ortalama bisiklet sayısı: {avg_not_sunny:.2f}")
print(f"Artış oranı: %{increase_ratio:.2f}")

# 6. Görselleştirme
labels = ['Güneşli Günler', 'Güneşsiz Günler']
means = [avg_sunny, avg_not_sunny]

plt.figure(figsize=(8,5))
plt.bar(labels, means, color=['orange', 'gray'])
plt.title('Güneşli vs Güneşsiz Günlerde Bisiklet Kullanımı')
plt.ylabel('Ortalama Bisiklet Sayısı')
plt.show()
