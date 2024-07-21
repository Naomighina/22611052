import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(r'C:\Kuliah\Semester4\MPML\UAS\transactions.csv')

# Periksa nilai hilang
print("Nilai hilang per kolom:")
print(data.isnull().sum())

# Tangani nilai hilang (ganti dengan strategi yang sesuai)
# Misalnya, mengisi kolom numerik dengan rata-rata atau median
# data['Amount (INR)'].fillna(data['Amount (INR)'].mean(), inplace=True)

# Untuk kolom kategorik, pertimbangkan imputasi modus atau membuat kategori baru

# Buat objek LabelEncoder
label_encoder = LabelEncoder()
for col in ['Transaction ID', 'Sender Name', 'Sender UPI ID', 'Receiver Name', 'Receiver UPI ID', 'Status']:
    data[col] = label_encoder.fit_transform(data[col])
data = data.drop(columns=['Timestamp'])

# Kodekan kolom kategorik
data['Sender Name'] = label_encoder.fit_transform(data['Sender Name'])
data['Sender UPI ID'] = label_encoder.fit_transform(data['Sender UPI ID'])
data['Receiver Name'] = label_encoder.fit_transform(data['Receiver Name'])
data['Receiver UPI ID'] = label_encoder.fit_transform(data['Receiver UPI ID'])
data['Status'] = label_encoder.fit_transform(data['Status'])

# Buat objek StandardScaler
scaler = StandardScaler()

# Skala fitur numerik (jika diperlukan)
# data['Amount (INR)'] = scaler.fit_transform(data[['Amount (INR)']])
print("\nTipe data setelah preprocessing:")
print(data.dtypes)

data_scaled = data

data['target'] = np.random.rand(len(data))