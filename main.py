import pandas as pd
from scipy.stats import alpha
import matplotlib.pyplot as plt
from scipy.version import git_revision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/seanryan/Downloads/EURUSD=X.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Create lagged features
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)
data = data.dropna()

X = data[['Lag1', 'Lag2', 'Lag3']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data['Date'].iloc[-len(y_test):], y_pred, label='Predicted', alpha=0.7)
plt.plot(data['Date'].iloc[-len(y_test):], y_test, label='Actual', alpha=0.7)
plt.title('EUR/USD Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print("Mean Squared Error:", mse)


