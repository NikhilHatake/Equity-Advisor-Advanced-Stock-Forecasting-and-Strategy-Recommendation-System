import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from prophet import Prophet

# Load data
train = pd.read_csv("/kaggle/input/ue21cs342aa2/train.csv")
test = pd.read_csv("/kaggle/input/ue21cs342aa2/test.csv")

# Display data overview and statistics
print("Data Overview:")
print(train.head())
print("\nSummary Statistics:")
print(train.describe())

# Convert 'Date' from string to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Feature engineering: extract day of the week and month from 'Date'
for df in [train, test]:
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

# Regression model to predict Close prices
X_train = train[['Open', 'Volume']]
y_train = train['Close']
regression_model = LinearRegression().fit(X_train, y_train)

X_test = test[['Open', 'Volume']]
test['Close'] = regression_model.predict(X_test)

# Prophet model for forecasting (optional detailed forecasting)
prophet_data = train.rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)

# Classification model to predict Strategy
features = ['Open', 'Volume', 'Day_of_Week', 'Month']
classification_model = RandomForestClassifier().fit(train[features], train['Strategy'])
test['Strategy'] = classification_model.predict(test[features])

# Prepare submission
submission = test[['id', 'Date', 'Close', 'Strategy']]
print(submission.head())
submission.to_csv('submission.csv', index=False)

# Plotting
train.hist(figsize=(12, 8))
plt.show()

correlation_matrix = train.drop(columns=['Date', 'Strategy']).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
