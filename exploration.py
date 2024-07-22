import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset (pastikan path file Anda disesuaikan)
data = pd.read_csv(r'C:\UASMPML\transactions.csv')

# Inspeksi Data
print(data.info())  # Periksa tipe data dan nilai hilang
print(data.describe())  # Statistik deskriptif untuk kolom numerik

# Analisis Univariat
# Variabel numerik
sns.histplot(data['Amount (INR)'], bins=30)
plt.show()
sns.boxplot(x=data['Amount (INR)'])
plt.show()

# Variabel kategorik
sns.countplot(x='Status', data=data)
plt.show()

# Analisis Bivariat
sns.scatterplot(x='Amount (INR)', y='Timestamp', data=data)
plt.show()
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Analisis Deret Waktu (jika berlaku)
# Konversi timestamp ke format datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# Kelompokkan data berdasarkan periode waktu (misalnya, hari, bulan, tahun) dan hitung agregat
# Visualisasikan menggunakan plot garis

# Pembersihan dan Praproses Data (jika perlu)
# Tangani nilai hilang (misalnya, imputasi, penghapusan)
# Tangani outlier (misalnya, capping, flooring, penghapusan)
# Atasi inkonsistensi dalam data (misalnya, konversi tipe data, pemformatan)