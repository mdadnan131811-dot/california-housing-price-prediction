# Housing Price Prediction - Data Science Project
# Author: Md Adnan

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("=== 1. Loading Dataset ===")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

print("\n=== 2. Exploratory Data Analysis (EDA) ===")
print("Missing values in dataset:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe().T)

print("\n=== 3. Data Preprocessing & Feature Engineering ===")
# Feature Matrix (X) and Target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== 4. Model Training (Random Forest Regressor) ===")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n=== 5. Model Evaluation ===")
predictions = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score (Model Accuracy): {r2 * 100:.2f}%")

print("\n=== 6. Feature Importance ===")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top Features Driving House Prices:\n", importances)
