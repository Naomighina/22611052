# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'C:\UASHEALTHMPML\Health index.csv')

# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())

# Handle missing values (if any)
# Example: Impute with mean
# data.fillna(data.mean(), inplace=True)

# Scaling numerical features
scaler = StandardScaler()
scaled_columns = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content']
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

print("\nData types after preprocessing:")
print(data.dtypes)

