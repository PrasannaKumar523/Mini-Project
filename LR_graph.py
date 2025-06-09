import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('cleaned_mumbai_dataset_new.csv').sample(500, random_state=42)

# Define the feature set and target
X = data[['bhk', 'type', 'locality', 'area', 'region', 'status', 'age']]
y = data['price']

# Encode categorical features
categorical_features = ['type', 'locality', 'region', 'status', 'age']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)


# Graph 1: Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Actual Prices')
plt.plot(y_test, y_test, color='red', linewidth=2, label='Fitted Line')
plt.title('Linear Regression Mode')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid()
plt.show()

