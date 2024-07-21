import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Kuliah/Semester4/MPML/UAS/transactions.csv')

# Encoding label
label_encoder = LabelEncoder()
for col in ['Transaction ID', 'Sender Name', 'Sender UPI ID', 'Receiver Name', 'Receiver UPI ID', 'Status']:
    data[col] = label_encoder.fit_transform(data[col])

# Drop kolom yang tidak diperlukan
data = data.drop(columns=['Timestamp'])

# Define features and target
X = data.drop('Status', axis=1)
y = data['Status']

# Display feature data types
print("\nFeature Data Types:")
print(X.dtypes)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
model_lr = LogisticRegression(random_state=42)
model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_dt = {'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10]}
param_grid_rf = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

grid_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, scoring='accuracy')
grid_dt = GridSearchCV(model_dt, param_grid_dt, cv=5, scoring='accuracy')
grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='accuracy')

# Fit models
grid_lr.fit(X_train, y_train)
grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

# Get best estimators
best_lr = grid_lr.best_estimator_
best_dt = grid_dt.best_estimator_
best_rf = grid_rf.best_estimator_

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    return accuracy, precision, recall, f1

# Evaluate models
print("Logistic Regression:")
acc_lr, prec_lr, rec_lr, f1_lr = evaluate_model(best_lr, X_test, y_test)
print("\nDecision Tree:")
acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(best_dt, X_test, y_test)
print("\nRandom Forest:")
acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(best_rf, X_test, y_test)

# Cross-validation
cv_scores_lr = cross_val_score(best_lr, X, y, cv=5, scoring='accuracy')
cv_scores_dt = cross_val_score(best_dt, X, y, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')

print("\nCross-validation scores (Logistic Regression):", cv_scores_lr)
print("Mean CV score (Logistic Regression):", np.mean(cv_scores_lr))
print("\nCross-validation scores (Decision Tree):", cv_scores_dt)
print("Mean CV score (Decision Tree):", np.mean(cv_scores_dt))
print("\nCross-validation scores (Random Forest):", cv_scores_rf)
print("Mean CV score (Random Forest):", np.mean(cv_scores_rf))

# Visualize model comparison
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [acc_lr, acc_dt, acc_rf]
precisions = [prec_lr, prec_dt, prec_rf]
recalls = [rec_lr, rec_dt, rec_rf]
f1_scores = [f1_lr, f1_dt, f1_rf]

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
sns.barplot(x=model_names, y=accuracies)
plt.title('Accuracy')

plt.subplot(2, 2, 2)
sns.barplot(x=model_names, y=precisions)
plt.title('Precision')

plt.subplot(2, 2, 3)
sns.barplot(x=model_names, y=recalls)
plt.title('Recall')

plt.subplot(2, 2, 4)
sns.barplot(x=model_names, y=f1_scores)
plt.title('F1-score')

plt.tight_layout()
plt.show()
