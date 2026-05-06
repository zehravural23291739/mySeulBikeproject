# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:49:06 2025

@author:
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gerçek test verisi ve tahminler (bunlar modelden gelmeli)
# y_test: gerçek değerler
# y_pred: modelin tahmin ettiği değerler
# X_test: test için kullanılan giriş verileri

# X_test DataFrame'ini birleştiriyoruz (tahminle karşılaştırmak için)
test_df = X_test.copy()
test_df['Gerçek'] = y_test
test_df['Tahmin'] = y_pred

# Güneşli günleri seç (tahmin setinde)
sunny = test_df[(test_df['Solar Radiation (MJ/m2)'] > 0.5) &
                (test_df['Rainfall(mm)'] == 0) &
                (test_df['Snowfall (cm)'] == 0)]

not_sunny = test_df.drop(sunny.index)

# Gerçek verilere göre ortalamalar
gercek_avg_sunny = sunny['Gerçek'].mean()
gercek_avg_notsunny = not_sunny['Gerçek'].mean()
gercek_artis = ((gercek_avg_sunny - gercek_avg_notsunny) / gercek_avg_notsunny) * 100

# Model tahminlerine göre ortalamalar
tahmin_avg_sunny = sunny['Tahmin'].mean()
tahmin_avg_notsunny = not_sunny['Tahmin'].mean()
tahmin_artis = ((tahmin_avg_sunny - tahmin_avg_notsunny) / tahmin_avg_notsunny) * 100

# Sonuçları yazdır
print(f"📊 Gerçek verilerde güneşli günlerde artış oranı: %{gercek_artis:.2f}")
print(f"🤖 Model tahminlerinde güneşli günlerde artış oranı: %{tahmin_artis:.2f}")

# Karşılaştırma grafiği
labels = ['Güneşli Günler', 'Güneşsiz Günler']
gercek_means = [gercek_avg_sunny, gercek_avg_notsunny]
tahmin_means = [tahmin_avg_sunny, tahmin_avg_notsunny]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, gercek_means, width, label='Gerçek', color='skyblue')
rects2 = ax.bar(x + width/2, tahmin_means, width, label='Tahmin', color='orange')

ax.set_ylabel('Ortalama Bisiklet Sayısı')
ax.set_title('Güneşli vs Güneşsiz Günlerde Kullanım (Gerçek vs Model)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
