import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Modeli Eğit ve Kaydet
def train_and_save_model():
    print("Model eğitiliyor ve kaydediliyor...")
    df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')

    # Tarihi doğru formatta işle
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop(['Date'], axis=1, inplace=True)

    y = df['Rented Bike Count']
    X = df.drop(['Rented Bike Count'], axis=1)

    # Kategorik sütunlar
    categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, 'bike_rental_model.pkl')
    print(" Model başarıyla eğitildi ve kaydedildi.")

# 🔮 Tahmin Fonksiyonu
def predict_bike_rentals():
    if not os.path.exists('bike_rental_model.pkl'):
        train_and_save_model()

    model = joblib.load('bike_rental_model.pkl')

    print("\n Bisiklet Tahmini İçin Bilgileri Giriniz:")
    year = int(input("Yıl: "))
    month = int(input("Ay: "))
    day = int(input("Gün: "))
    hour = int(input("Saat (0-23): "))
    temp = float(input("Sıcaklık (°C): "))
    humidity = float(input("Nem (%): "))
    wind = float(input("Rüzgar Hızı (m/s): "))
    visibility = float(input("Görüş Mesafesi (10m): "))
    dew_temp = float(input("Çiy Noktası Sıcaklığı (°C): "))
    solar = float(input("Güneş Radyasyonu (MJ/m2): "))
    rain = float(input("Yağış (mm): "))
    snow = float(input("Kar (cm): "))
    season = input("Mevsim (Spring/Summer/Autumn/Winter): ")
    holiday = input("Tatil mi? (Yes/No): ")
    working_day = input("İş Günü mü? (Yes/No): ")

    user_input = pd.DataFrame([{
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
        'Functioning Day': working_day,
        'year': year,
        'month': month,
        'day': day
    }])

    prediction = model.predict(user_input)[0]
    print(f"\n Tahmini Kiralanan Bisiklet Sayısı: {int(prediction)} adet")


if __name__ == "__main__":
    predict_bike_rentals()
