# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:42:32 2025

@author:
"""

import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load('bike_rental_model.pkl')

st.title(" Bisiklet Kiralama Tahmin Uygulaması")
st.markdown("Seul şehri için saatlik bisiklet kiralama sayısını tahmin edin.")

# Kullanıcı girdileri
year = st.number_input("Yıl", min_value=2017, max_value=2025, value=2018)
month = st.selectbox("Ay", list(range(1,13)))
day = st.selectbox("Gün", list(range(1,32)))
hour = st.slider("Saat", 0, 23, 12)

temp = st.number_input("Sıcaklık (°C)", value=15.0)
humidity = st.number_input("Nem (%)", value=50.0)
wind = st.number_input("Rüzgar Hızı (m/s)", value=2.0)
visibility = st.number_input("Görüş Mesafesi (10m)", value=2000.0)
dew_temp = st.number_input("Çiy Noktası (°C)", value=10.0)
solar = st.number_input("Güneş Radyasyonu (MJ/m2)", value=0.5)
rain = st.number_input("Yağış (mm)", value=0.0)
snow = st.number_input("Kar (cm)", value=0.0)

season = st.selectbox("Mevsim", ["Spring", "Summer", "Autumn", "Winter"])
holiday = st.selectbox("Tatil mi?", ["No", "Yes"])
functioning_day = st.selectbox("İş Günü mü?", ["Yes", "No"])

# Tahmin butonu
if st.button("Tahmini Kiralanan Bisiklet Sayısını Göster"):
    input_df = pd.DataFrame([{
        'Hour': hour,
        'Temperature(°C)': temp,
        'Humidity(%)': humidity,
        'Wind speed (m/s)': wind,
        'Visibility (10m)': visibility,
        'Dew point temperature(°C)': dew_temp,
        'Solar Radiation (MJ/m2)': solar,
        'Rainfall(mm)': rain,
        'Snowfall (cm)': snow,
        'Seasons': season,
        'Holiday': holiday,
        'Functioning Day': functioning_day,
        'year': year,
        'month': month,
        'day': day
    }])

    # Veri dönüştürme adımını manuel yapıyorsanız
transformed_input = transformer.transform(input_df)  # input_df'yi dönüştür
prediction = model.predict(transformed_input)

st.success(f" Tahmini Kiralanan Bisiklet Sayısı: **{int(prediction)}** adet")
