import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Analisis Data Transaksi')

# Muat dataset
data = pd.read_csv('transactions.csv')

# Tampilkan informasi dasar tentang dataset
st.write(data.info())
st.write(data.describe())

# Analisis Univariat
st.header('Analisis Univariat')
st.subheader('Variabel Numerik')
fig, ax = plt.subplots()
sns.histplot(data['Amount (INR)'], bins=30, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x=data['Amount (INR)'], ax=ax)
st.pyplot(fig)

st.subheader('Variabel Kategorik')
fig, ax = plt.subplots()
sns.countplot(x='Status', data=data, ax=ax)
st.pyplot(fig)

# Analisis Bivariat
st.header('Analisis Bivariat')
fig, ax = plt.subplots()
sns.scatterplot(x='Amount (INR)', y='Timestamp', data=data, ax=ax)
st.pyplot(fig)

# Memilih hanya kolom numerik
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Menghitung matriks korelasi
correlation_matrix = numerical_data.corr()

# Menampilkan matriks korelasi
print(correlation_matrix)
