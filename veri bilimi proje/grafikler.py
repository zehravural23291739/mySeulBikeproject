# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:42:35 2025

@author: vural
"""

# -- coding: utf-8 --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri setini yükle
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')

# 2. Tarih verisini işle
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df.drop('Date', axis=1)

# 3. Kategorik dönüşümler
df['Seasons'] = df['Seasons'].astype('category')
df['Holiday'] = df['Holiday'].astype('category')
df['Functioning Day'] = df['Functioning Day'].astype('category')
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# 4. Grafikler

# A. Ay ay bisiklet kiralama sayıları
plt.figure(figsize=(10,6))
sns.barplot(x='Month', y='BikeCount', data=df, ci=None, estimator=np.mean)
plt.title('Aylara Göre Ortalama Bisiklet Kiralama')
plt.xlabel('Ay')
plt.ylabel('Ortalama Kiralama')
plt.tight_layout()
plt.savefig("aylik_kiralama.png")
plt.close()

# B. Hava sıcaklığına göre kiralama
plt.figure(figsize=(10,6))
sns.scatterplot(x='Temperature(°C)', y='BikeCount', data=df, alpha=0.5)
plt.title('Sıcaklık ve Bisiklet Kiralama İlişkisi')
plt.xlabel('Sıcaklık (°C)')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("sicaklik_kiralama.png")
plt.close()

# C. Mevsimlere göre kiralama
plt.figure(figsize=(10,6))
sns.boxplot(x='Seasons', y='BikeCount', data=df)
plt.title('Mevsimlere Göre Bisiklet Kiralama Dağılımı')
plt.xlabel('Mevsim')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("mevsim_kiralama.png")
plt.close()

# D. Haftanın gününe göre kiralama
plt.figure(figsize=(10,6))
sns.boxplot(x='DayOfWeek', y='BikeCount', data=df)
plt.title('Haftanın Günlerine Göre Bisiklet Kiralama')
plt.xlabel('Gün (0 = Pazartesi)')
plt.ylabel('Kiralama')
plt.tight_layout()
plt.savefig("gunluk_kiralama.png")
plt.close()

# E. Korelasyon matrisi
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Sayısal Değişkenler Korelasyon Matrisi')
plt.tight_layout()
plt.savefig("korelasyon_matrisi.png")
plt.close()

from scipy.stats import ttest_ind

# 1. Grupları ayır
holiday_group = df[df['Holiday'] == 'Holiday']['BikeCount']
no_holiday_group = df[df['Holiday'] == 'No Holiday']['BikeCount']

# 2. T-testi uygulama
t_stat, p_value = ttest_ind(holiday_group, no_holiday_group, equal_var=False)

# 3. Sonuçları yazdır
print("\nHipotez Testi Sonuçları (Tatil vs. Tatil Değil):")
print(f"T-istatistiği: {t_stat:.3f}")
print(f"P-değeri: {p_value:.5f}")

# 4. Sonuç grafiği (bar plot + p-değeri etiketi)
plt.figure(figsize=(8, 6))
sns.barplot(x='Holiday', y='BikeCount', data=df, ci='sd', palette='Set2')
plt.title('Tatil Günlerinde Kiralama Ortalaması')
plt.xlabel('Gün Türü')
plt.ylabel('Bisiklet Kiralama')

# P-değerini grafiğe ekle
plt.text(0.5, df['BikeCount'].max() * 0.9, f"p-değeri = {p_value:.4f}", 
         ha='center', fontsize=12, color='red')

plt.tight_layout()
plt.savefig("hipotez_tatil_karsilastirmasi.png")
plt.close()
# Ortalama kiralama sayıları
mean_holiday = holiday_group.mean()
mean_no_holiday = no_holiday_group.mean()

# Yüzde değişim hesabı
percent_change = ((mean_holiday - mean_no_holiday) / mean_no_holiday) * 100

# Sonucu yazdır
print(f"Tatil günlerindeki ortalama kiralama: {mean_holiday:.2f}")
print(f"Tatil olmayan günlerdeki ortalama kiralama: {mean_no_holiday:.2f}")
print(f"Yüzde değişim: %{percent_change:.2f}")

print("Tüm grafikler oluşturuldu ve kaydedildi.")

plt.figure(figsize=(10,6))
sns.scatterplot(x='Humidity(%)', y='BikeCount', data=df, alpha=0.5)
plt.title('Nem ve Bisiklet Kiralama İlişkisi')
plt.xlabel('Nem (%)')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("nem_kiralama.png")
plt.close()

plt.figure(figsize=(10,6))
sns.lineplot(x='Hour', y='BikeCount', data=df, estimator=np.mean, ci=None)
plt.title('Günün Saatine Göre Ortalama Kiralama')
plt.xlabel('Saat')
plt.ylabel('Ortalama Kiralama')
plt.tight_layout()
plt.savefig("saatlik_kiralama.png")
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(x='Functioning Day', y='BikeCount', data=df, palette='pastel')
plt.title('Fonksiyonel Günlerde Kiralama Dağılımı')
plt.xlabel('Fonksiyonel Gün')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("fonksiyonel_kiralama.png")
plt.close()


plt.figure(figsize=(10,6))
sns.scatterplot(x='Rainfall(mm)', y='BikeCount', data=df, alpha=0.5)
plt.title('Yağış ve Kiralama İlişkisi')
plt.xlabel('Yağış (mm)')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("yagis_kiralama.png")
plt.close()


# Eğer aylar string olarak geliyorsa önce sayıya çevir:
month_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

plt.figure(figsize=(10,6))
sns.barplot(x='Month', y='BikeCount', data=df, estimator=np.mean, ci=None, order=month_order)
plt.title('Aylara Göre Ortalama Bisiklet Kiralama')
plt.xlabel('Ay')
plt.ylabel('Ortalama Kiralama')
plt.tight_layout()
plt.savefig("duzeltilmis_aylik_kiralama.png")
plt.close()



plt.figure(figsize=(10,6))
sns.scatterplot(x='Wind speed (m/s)', y='BikeCount', data=df, alpha=0.5)
plt.title('Rüzgar Hızı ve Kiralama İlişkisi')
plt.xlabel('Rüzgar Hızı (m/s)')
plt.ylabel('Bisiklet Kiralama')
plt.tight_layout()
plt.savefig("ruzgar_kiralama.png")
plt.close()


plt.figure(figsize=(10,6))
sns.scatterplot(x='Temperature(°C)', y='BikeCount', data=df)
plt.title('Sıcaklık vs Kiralama Sayısı')
plt.xlabel('Sıcaklık (°C)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("sicaklik_vs_kiralama.png")
plt.close()


plt.figure(figsize=(10,6))
sns.boxplot(x='Functioning Day', y='BikeCount', data=df)
plt.title('İş Günü Durumuna Göre Kiralama')
plt.xlabel('İş Günü mü?')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("is_gunu_kiralama.png")
plt.close()



df['Date'] = pd.to_datetime(df['Date'])
daily_mean = df.groupby('Date')['BikeCount'].mean().reset_index()

plt.figure(figsize=(15,6))
sns.lineplot(x='Date', y='BikeCount', data=daily_mean)
plt.title('Günlük Ortalama Kiralama Zaman Serisi')
plt.xlabel('Tarih')
plt.ylabel('Ortalama Kiralama')
plt.tight_layout()
plt.savefig("zaman_serisi_kiralama.png")
plt.close()



plt.figure(figsize=(10,6))
sns.scatterplot(x='Humidity(%)', y='BikeCount', data=df, alpha=0.5)
plt.title('Nem vs Kiralama Sayısı')
plt.xlabel('Nem (%)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("nem_vs_kiralama.png")
plt.close()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.barplot(x='DayOfWeek', y='Rented Bike Count', data=df, estimator='mean', ci=None)
plt.xticks(ticks=range(7), labels=['Pzt', 'Salı', 'Çar', 'Per', 'Cuma', 'Cmt', 'Paz'])
plt.title('Haftanın Günlerine Göre Ortalama Kiralama')
plt.ylabel('Ortalama Kiralama Sayısı')
plt.xlabel('Gün')
plt.tight_layout()
plt.savefig("hafta_gunlerine_gore_kiralama.png")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x='Hour', y='Rented Bike Count', data=df)
plt.title('Saatlik Kiralama Dağılımı')
plt.xlabel('Saat')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("saatlik_kiralama_dagilimi.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Temperature(°C)', y='Rented Bike Count', data=df, alpha=0.5)
plt.title('Sıcaklık ile Kiralama İlişkisi')
plt.xlabel('Sıcaklık (°C)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("sicaklik_kiralama_iliski.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Humidity(%)', y='Rented Bike Count', data=df, alpha=0.5)
plt.title('Nem ile Kiralama İlişkisi')
plt.xlabel('Nem (%)')
plt.ylabel('Kiralama Sayısı')
plt.tight_layout()
plt.savefig("nem_kiralama_iliski.png")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Weather', y='Rented Bike Count', data=df, estimator='mean', ci=None)
plt.title('Hava Durumuna Göre Ortalama Kiralama')
plt.ylabel('Ortalama Kiralama')
plt.xlabel('Hava Durumu')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("hava_durumuna_gore_kiralama.png")
plt.show()

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')

# Tarih işlemleri
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})

# Mevsimlere göre aylık ortalama kiralama hesapla
seasonal_avg = df.groupby(['Seasons', 'Month'])['BikeCount'].mean().unstack().T

# Mevsim sıralamasını düzenle (İlkbahar, Yaz, Sonbahar, Kış)
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
seasonal_avg = seasonal_avg[season_order]

# Ay isimleri için mapping
month_names = {
    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 
    5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
    9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
}

# Grafik oluşturma
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
ax = seasonal_avg.plot(kind='bar', width=0.8, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])

plt.title('Mevsimlere Göre Aylık Ortalama Bisiklet Kiralama', fontsize=16, pad=20)
plt.xlabel('Ay', fontsize=12)
plt.ylabel('Ortalama Kiralama Sayısı', fontsize=12)
plt.xticks(range(12), [month_names[m] for m in range(1,13)], rotation=45, ha='right')
plt.legend(title='Mevsim', bbox_to_anchor=(1.05, 1), loc='upper left')

# Çubukların üzerine değerleri yazma
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points',
                fontsize=9)

plt.tight_layout()
plt.savefig("mevsimlik_aylik_kiralama_grafik.png", dpi=300, bbox_inches='tight')
plt.close()

# Tablo oluşturma
plt.figure(figsize=(12, 6))
plt.axis('off')

# Tablo başlığı
plt.title('Mevsimlere Göre Aylık Ortalama Kiralama', y=1.08, fontsize=14)

# Tablo içeriği
table = plt.table(cellText=seasonal_avg.round(0).astype(int).values,
                 colLabels=seasonal_avg.columns,
                 rowLabels=[month_names[m] for m in seasonal_avg.index],
                 loc='center',
                 cellLoc='center',
                 colColours=['#f7f7f7']*4)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig("mevsimlik_aylik_kiralama_tablo.png", dpi=300, bbox_inches='tight')
plt.close()

# Verileri CSV olarak kaydet
seasonal_avg.round(0).astype(int).to_csv('mevsimlik_aylik_kiralama.csv', encoding='utf-8-sig')

print("İşlemler başarıyla tamamlandı. Grafik ve tablolar kaydedildi.")