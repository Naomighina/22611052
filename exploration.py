# exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\UASHEALTHMPML\Health index.csv')

# Inspect the data
print(data.info())
print(data.describe())

# Univariate Analysis
# Numerical Variables
sns.histplot(data['Health_index'], bins=30)
plt.title('Distribution of Health Index')
plt.show()

sns.boxplot(x=data['Health_index'])
plt.title('Boxplot of Health Index')
plt.show()

# Bivariate Analysis
sns.scatterplot(x='Health_index', y='Life_expectation', data=data)
plt.title('Health Index vs Life Expectation')
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()
