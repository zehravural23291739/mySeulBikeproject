# -- coding: utf-8 --
"""
Seoul Bisiklet Kiralama Tahmin Projesi - Konsol Çıktılı Tam Sürüm
"""

# 1. Gerekli Kütüphanelerin Yüklenmesi
print("1. Gerekli kütüphaneler yükleniyor...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 2. Veri Yükleme
print("\n2. Veri seti yükleniyor...")
try:
    df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
    print("Veri seti başarıyla yüklendi!")
    print(f"\nVeri seti boyutu: {df.shape}")
    print("\nİlk 5 kayıt:")
    print(df.head())
except Exception as e:
    print(f"Hata oluştu: {e}")
    exit()

# 3. Veri Ön İşleme
print("\n3. Veri ön işleme yapılıyor...")


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# Tarih işleme
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df.drop('Date', axis=1)

# Kategorik değişkenler
df['Seasons'] = df['Seasons'].astype('category')
df['Holiday'] = df['Holiday'].astype('category')
df['Functioning Day'] = df['Functioning Day'].astype('category')
df = df.rename(columns={'Rented Bike Count': 'BikeCount'})

# Eksik veri kontrolü
print("\nEksik veri kontrolü:")
print(df.isnull().sum())

# 4. Veri Analizi (Konsol Çıktıları)
print("\n4. Veri analizi yapılıyor...")

print("\nTemel istatistikler:")
print(df.describe())

print("\nKategorik değişkenlerin dağılımı:")
print(df['Seasons'].value_counts())
print(df['Holiday'].value_counts())
print(df['Functioning Day'].value_counts())

# 5. Model Hazırlığı
print("\n5. Model için veri hazırlanıyor...")

X = df.drop('BikeCount', axis=1)
y = df['BikeCount']

categorical_cols = X.select_dtypes(include=['category']).columns
numerical_cols = X.select_dtypes(include=[np.number]).columns

# Ön işleme
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Veri bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 6. Model Eğitimi
print("\n6. Model eğitiliyor...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))])

model.fit(X_train, y_train)
print("Model eğitimi tamamlandı!")

# 7. Model Değerlendirme
print("\n7. Model değerlendiriliyor...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("R² Skoru:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

#ben yazdım 
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

print("\nModel Performans Metrikleri:")
print(f"- Ortalama Kare Hata (MSE): {mse:.2f}")
print(f"- Kök Ortalama Kare Hata (RMSE): {rmse:.2f}")
print(f"- Ortalama Mutlak Hata (MAE): {mae:.2f}")
print(f"- Açıklanan Varyans (R²): {r2:.2f}")

# 8. Örnek Tahmin
print("\n8. Örnek tahminler:")
sample_data = X_test.head(3)
sample_pred = model.predict(sample_data)
print("\nÖrnek veri:")
print(sample_data)
print("\nGerçek Değerler:")
print(y_test.head(3).values)
print("Tahminler:")
print(sample_pred)

# 9. Önemli Özellikler
print("\n9. Önemli özellikler:")
try:
    feature_importances = model.named_steps['regressor'].feature_importances_
    feature_names = numerical_cols.tolist()
    
    # Kategorik özellik isimlerini al
    ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names.extend(cat_features)
    
    # Önem sırasına göre sırala
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
    
    print("\nEn önemli 10 özellik:")
    print(importance_df)

    # Grafikle gösterim
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title("En Önemli 10 Özellik")
    plt.xlabel("Önem Skoru")
    plt.ylabel("Özellik")
    plt.tight_layout()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title("Gerçek vs Tahmin")
    print("R² Skoru:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    


# y_test: gerçek değerler
# y_pred: modelin tahmin ettiği değerler

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title("Gerçek vs Tahmin Edilen Değerler")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Doğruluk çizgisi
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Özellik önemleri alınırken hata: {e}")
joblib.dump(model, "bike_model.pkl")

print("\nProje başarıyla tamamlandı!")
