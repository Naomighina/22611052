# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
data = pd.read_csv('Health index.csv')


# Title
st.title('Health Index Prediction')

# Sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    Hydrogen = st.sidebar.number_input('Hydrogen', value=0.0)
    Oxygen = st.sidebar.number_input('Oxygen', value=0.0)
    Nitrogen = st.sidebar.number_input('Nitrogen', value=0.0)
    Methane = st.sidebar.number_input('Methane', value=0.0)
    CO = st.sidebar.number_input('CO', value=0.0)
    CO2 = st.sidebar.number_input('CO2', value=0.0)
    Ethylene = st.sidebar.number_input('Ethylene', value=0.0)
    Ethane = st.sidebar.number_input('Ethane', value=0.0)
    Acetylene = st.sidebar.number_input('Acetylene', value=0.0)
    DBDS = st.sidebar.number_input('DBDS', value=0.0)
    
    data = {
        'Hydrogen': Hydrogen,
        'Oxygen': Oxygen,
        'Nitrogen': Nitrogen,
        'Methane': Methane,
        'CO': CO,
        'CO2': CO2,
        'Ethylene': Ethylene,
        'Ethane': Ethane,
        'Acetylene': Acetylene,
        'DBDS': DBDS
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input features
st.subheader('User Input Features')
st.write(df)

# Initialize models
model_ridge = Ridge(random_state=42)
model_lasso = Lasso(random_state=42)
model_dt = DecisionTreeRegressor(random_state=42)
model_rf = RandomForestRegressor(random_state=42)


