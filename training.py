# training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
data = pd.read_csv(r'C:\UASHEALTHMPML\Health index.csv')

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
grid_ridge.fit(X_train, y_train)
grid_lasso.fit(X_train, y_train)
grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

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
    print('Mean Squared Error:', mse)
    print('R^2 Score:', r2)
    return mse, r2

# Evaluate models
print("\nRidge Regression:")
mse_ridge, r2_ridge = evaluate_model(best_ridge, X_test, y_test)
print("\nLasso Regression:")
mse_lasso, r2_lasso = evaluate_model(best_lasso, X_test, y_test)
print("\nDecision Tree Regressor:")
mse_dt, r2_dt = evaluate_model(best_dt, X_test, y_test)
print("\nRandom Forest Regressor:")
mse_rf, r2_rf = evaluate_model(best_rf, X_test, y_test)

# Cross-validation
cv_scores_ridge = cross_val_score(best_ridge, X, y, cv=5, scoring='r2')
cv_scores_lasso = cross_val_score(best_lasso, X, y, cv=5, scoring='r2')
cv_scores_dt = cross_val_score(best_dt, X, y, cv=5, scoring='r2')
cv_scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='r2')

print("\nCross-validation scores (Ridge Regression):", cv_scores_ridge)
print("Mean CV score (Ridge Regression):", np.mean(cv_scores_ridge))
print("\nCross-validation scores (Lasso Regression):", cv_scores_lasso)
print("Mean CV score (Lasso Regression):", np.mean(cv_scores_lasso))
print("\nCross-validation scores (Decision Tree):", cv_scores_dt)
print("Mean CV score (Decision Tree):", np.mean(cv_scores_dt))
print("\nCross-validation scores (Random Forest):", cv_scores_rf)
print("Mean CV score (Random Forest):", np.mean(cv_scores_rf))

# Visualize model comparison
model_names = [ 'Ridge Regression', 'Lasso Regression', 'Decision Tree', 'Random Forest']
r2_scores = [r2_ridge, r2_lasso, r2_dt, r2_rf]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=r2_scores)
plt.title('R^2 Scores')
plt.ylabel('R^2 Score')
plt.show()
