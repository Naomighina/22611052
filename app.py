# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Health Index Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Health index.csv')
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page", ["Exploration", "Preprocessing", "Model Training"])

# Exploration Page
if options == "Exploration":
    st.title("Exploration and Analysis of Health Index Dataset")

    # Show data
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    # Data Info
    st.subheader("Data Info")
    buffer = st.empty()
    buffer.info = data.info()

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    st.text("Distribution of Health Index")
    fig, ax = plt.subplots()
    sns.histplot(data['Health_index'], bins=30, ax=ax)
    st.pyplot(fig)

    st.text("Boxplot of Health Index")
    fig, ax = plt.subplots()
    sns.boxplot(x=data['Health_index'], ax=ax)
    st.pyplot(fig)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    st.text("Health Index vs Life Expectation")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Health_index', y='Life_expectation', data=data, ax=ax)
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, ax=ax)
    st.pyplot(fig)

# Preprocessing Page
elif options == "Preprocessing":
    st.title("Preprocessing the Health Index Dataset")

    # Check for missing values
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    # Handle missing values (if any)
    st.subheader("Impute Missing Values")
    st.text("Filling missing values with the mean of each column...")
    data.fillna(data.mean(), inplace=True)
    st.text("Missing values have been filled.")

    # Scaling numerical features
    st.subheader("Feature Scaling")
    scaler = StandardScaler()
    scaled_columns = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    st.write(data.head())

    # Save preprocessed data for model training
    preprocessed_file = "preprocessed_health_index.csv"
    data.to_csv(preprocessed_file, index=False)
    st.text(f"Preprocessed data saved as {preprocessed_file}")

# Model Training Page
elif options == "Model Training":
    st.title("Model Training and Evaluation")

    # Load preprocessed dataset
    data = pd.read_csv("preprocessed_health_index.csv")

    # Define features and target
    X = data.drop('Health_index', axis=1)
    y = data['Health_index']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    with st.spinner('Training models...'):
        grid_ridge.fit(X_train, y_train)
        grid_lasso.fit(X_train, y_train)
        grid_dt.fit(X_train, y_train)
        grid_rf.fit(X_train, y_train)

    st.success('Models trained successfully!')

    # Get best estimators
    best_ridge = grid_ridge.best_estimator_
    best_lasso = grid_lasso.best_estimator_
    best_dt = grid_dt.best_estimator_
    best_rf = grid_rf.best_estimator_

    # Evaluate models
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    # Show evaluation results
    st.subheader("Model Evaluation Results")

    mse_ridge, r2_ridge = evaluate_model(best_ridge, X_test, y_test)
    st.write(f"**Ridge Regression**: MSE = {mse_ridge:.3f}, R^2 = {r2_ridge:.3f}")

    mse_lasso, r2_lasso = evaluate_model(best_lasso, X_test, y_test)
    st.write(f"**Lasso Regression**: MSE = {mse_lasso:.3f}, R^2 = {r2_lasso:.3f}")

    mse_dt, r2_dt = evaluate_model(best_dt, X_test, y_test)
    st.write(f"**Decision Tree Regressor**: MSE = {mse_dt:.3f}, R^2 = {r2_dt:.3f}")

    mse_rf, r2_rf = evaluate_model(best_rf, X_test, y_test)
    st.write(f"**Random Forest Regressor**: MSE = {mse_rf:.3f}, R^2 = {r2_rf:.3f}")

    # Cross-validation
    st.subheader("Cross-validation Scores")
    cv_scores_ridge = cross_val_score(best_ridge, X, y, cv=5, scoring='r2')
    cv_scores_lasso = cross_val_score(best_lasso, X, y, cv=5, scoring='r2')
    cv_scores_dt = cross_val_score(best_dt, X, y, cv=5, scoring='r2')
    cv_scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='r2')

    st.write(f"**Ridge Regression CV scores**: {cv_scores_ridge}, Mean CV score = {np.mean(cv_scores_ridge):.3f}")
    st.write(f"**Lasso Regression CV scores**: {cv_scores_lasso}, Mean CV score = {np.mean(cv_scores_lasso):.3f}")
    st.write(f"**Decision Tree CV scores**: {cv_scores_dt}, Mean CV score = {np.mean(cv_scores_dt):.3f}")
    st.write(f"**Random Forest CV scores**: {cv_scores_rf}, Mean CV score = {np.mean(cv_scores_rf):.3f}")

    # Visualize model comparison
    st.subheader("Model Comparison")
    model_names = ['Ridge Regression', 'Lasso Regression', 'Decision Tree', 'Random Forest']
    r2_scores = [r2_ridge, r2_lasso, r2_dt, r2_rf]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model_names, y=r2_scores, ax=ax)
    ax.set_title('R^2 Scores Comparison')
    ax.set_ylabel('R^2 Score')
    st.pyplot(fig)
