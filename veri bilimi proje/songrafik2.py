# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:44 2025

@author:
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Veri yükleme ve ön işleme
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Season'] = df['Seasons'].astype('category')
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})

# 1. Isı Haritası - Saatlik ve Aylık Kiralama
plt.figure(figsize=(12, 8))
hour_month = df.pivot_table(index='Hour', columns='Month', values='BikeCount', aggfunc='mean')
sns.heatmap(hour_month, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=.5)
plt.title('Saatlik ve Aylık Ortalama Kiralama Isı Haritası', pad=20)
plt.tight_layout()
plt.savefig("heatmap_hour_month.png", dpi=300)
plt.close()

# 2. Alternatif: Stacked Bar Chart (Treemap yerine)
plt.figure(figsize=(12, 6))
season_month = df.groupby(['Season', 'Month'])['BikeCount'].mean().unstack()
season_month.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Mevsimlere Göre Aylık Kiralama (Stacked Bar Chart)')
plt.ylabel('Ortalama Kiralama Sayısı')
plt.tight_layout()
plt.savefig("stacked_bar_season_month.png", dpi=300)
plt.close()

# 3. Violin Plot ile Saatlik Dağılımlar
plt.figure(figsize=(14, 6))
sns.violinplot(x='Hour', y='BikeCount', data=df, palette='coolwarm', inner='quartile')
plt.title('Saatlere Göre Kiralama Dağılımı (Violin Plot)')
plt.savefig("violin_hourly.png", dpi=300)
plt.close()

# 4. Bubble Chart - Sıcaklık, Nem ve Kiralama
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='Temperature(°C)', y='Humidity(%)', 
                         size='BikeCount', hue='Season',
                         sizes=(20, 200), alpha=0.7, palette='viridis',
                         data=df.sample(1000))
plt.title('Sıcaklık, Nem ve Kiralama İlişkisi (Bubble Chart)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("bubble_temp_humidity.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Small Multiples - Haftanın Günlerine Göre Saatlik Kiralama
g = sns.FacetGrid(df, col='DayOfWeek', col_wrap=4, height=3, aspect=1.2)
g.map(sns.lineplot, 'Hour', 'BikeCount', ci=None, color='#4e79a7')
g.set_titles("Gün {col_name}")
g.set_xticks(range(0, 24, 4))
g.fig.suptitle('Haftanın Günlerine Göre Saatlik Kiralama Desenleri', y=1.05)
plt.tight_layout()
plt.savefig("small_multiples_hourly.png", dpi=300)
plt.close()

print("Tüm grafikler başarıyla oluşturuldu ve kaydedildi.")
