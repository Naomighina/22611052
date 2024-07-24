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

# Hyperparameter tuning using GridSearchCV
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_dt = {'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10]}
param_grid_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

grid_ridge = GridSearchCV(model_ridge, param_grid_ridge, cv=5, scoring='r2')
grid_lasso = GridSearchCV(model_lasso, param_grid_lasso, cv=5, scoring='r2')
grid_dt = GridSearchCV(model_dt, param_grid_dt, cv=5, scoring='r2')
grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='r2')

# Fit models
grid_ridge.fit(X_train, y_train)
grid_lasso.fit(X_train, y_train)
grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

# Get best estimators
best_ridge = grid_ridge.best_estimator_
best_lasso = grid_lasso.best_estimator_
best_dt = grid_dt.best_estimator_
best_rf = grid_rf.best_estimator_

# Display predictions
st.subheader('Predictions')
st.write('Ridge Regression:', pred_ridge[0])
st.write('Lasso Regression:', pred_lasso[0])
st.write('Decision Tree:', pred_dt[0])
st.write('Random Forest:', pred_rf[0])
