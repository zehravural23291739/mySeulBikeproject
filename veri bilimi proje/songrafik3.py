# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:48:49 2025

@author:
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
from statsmodels.graphics.mosaicplot import mosaic
import squarify

# Veri yükleme ve ön işleme
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Season'] = df['Seasons'].astype('category')
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})

# 1. Interaktif Isı Haritası (Plotly)
plt.figure(figsize=(12, 8))
hour_month = df.pivot_table(index='Hour', columns='Month', values='BikeCount', aggfunc='mean')
sns.heatmap(hour_month, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=.5)
plt.title('Saatlik ve Aylık Ortalama Kiralama Isı Haritası', pad=20)
plt.tight_layout()
plt.savefig("heatmap_hour_month.png", dpi=300)
plt.close()

# 2. Treemap ile Mevsim ve Ay Dağılımı
plt.figure(figsize=(12, 8))
season_month = df.groupby(['Season', 'Month'])['BikeCount'].mean().reset_index()
squarify.plot(sizes=season_month['BikeCount'], 
              label=[f"{row['Season']}\nAy:{row['Month']}\n{row['BikeCount']:.0f}" 
                     for _, row in season_month.iterrows()],
              alpha=0.8, color=sns.color_palette('pastel', len(season_month)))
plt.title('Mevsim ve Aylara Göre Kiralama Dağılımı (Treemap)')
plt.axis('off')
plt.tight_layout()
plt.savefig("treemap_season_month.png", dpi=300)
plt.close()

# 3. Violin Plot ile Saatlik Dağılımlar
plt.figure(figsize=(14, 6))
sns.violinplot(x='Hour', y='BikeCount', data=df, palette='coolwarm', inner='quartile')
plt.title('Saatlere Göre Kiralama Dağılımı (Violin Plot)')
plt.savefig("violin_hourly.png", dpi=300)
plt.close()

# 4. Radar Chart (Polar Plot) - Mevsimsel Örüntüler
def create_radar_chart():
    categories = ['00-03', '04-07', '08-11', '12-15', '16-19', '20-23']
    df['TimeCategory'] = pd.cut(df['Hour'], bins=[0,4,8,12,16,20,24], labels=categories)
    
    season_time = df.groupby(['Season', 'TimeCategory'])['BikeCount'].mean().unstack()
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for season in season_time.index:
        values = season_time.loc[season].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=season)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=categories)
    
    ax.set_title('Gün İçi Zaman Dilimlerine Göre Mevsimsel Kiralama Örüntüleri', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("radar_seasonal_patterns.png", dpi=300)
    plt.close()

create_radar_chart()

# 5. Animasyonlu Zaman Serisi (Plotly HTML çıktısı)
fig = px.line(df.groupby('Date')['BikeCount'].mean().reset_index(), 
              x='Date', y='BikeCount', 
              title='Günlük Bisiklet Kiralama Zaman Serisi',
              template='plotly_dark')
fig.write_html("animated_time_series.html")

# 6. Bubble Chart - Sıcaklık, Nem ve Kiralama
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

# 7. Small Multiples - Haftanın Günlerine Göre Saatlik Kiralama
g = sns.FacetGrid(df, col='DayOfWeek', col_wrap=4, height=3, aspect=1.2)
g.map(sns.lineplot, 'Hour', 'BikeCount', ci=None, color='#4e79a7')
g.set_titles("Gün {col_name}")
g.set_xticks(range(0, 24, 4))
g.fig.suptitle('Haftanın Günlerine Göre Saatlik Kiralama Desenleri', y=1.05)
plt.tight_layout()
plt.savefig("small_multiples_hourly.png", dpi=300)
plt.close()

# 8. Mosaic Plot - Mevsim ve Çalışma Günü Etkileşimi
plt.figure(figsize=(10, 8))
mosaic(df, ['Season', 'Functioning Day'], title='Mevsim ve Çalışma Günü Etkileşimi',
       properties=lambda key: {'color': 'r' if 'Yes' in key else 'b'}, gap=0.02)
plt.savefig("mosaic_season_functioning.png", dpi=300)
plt.close()

print("Tüm yenilikçi grafikler başarıyla oluşturuldu ve kaydedildi.")