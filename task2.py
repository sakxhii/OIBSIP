import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

file_path = 'dataset/Unemployment in India.csv'
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
data['Date'] = data['Date'].str.strip()
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data = data.dropna()

encoded_data = pd.get_dummies(data, columns=['Region', 'Frequency', 'Area'], drop_first=True)

X = encoded_data.drop(['Date', 'Estimated Unemployment Rate (%)'], axis=1)
y = encoded_data['Estimated Unemployment Rate (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print("Model training and evaluation completed.")

try:
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.title('Actual vs Predicted Unemployment Rate')
    plt.xlabel('Actual Rate')
    plt.ylabel('Predicted Rate')
    plt.show()
except Exception as e:
    print(f"Plotting failed: {e}")
finally:
    print("Model training and evaluation completed.")
