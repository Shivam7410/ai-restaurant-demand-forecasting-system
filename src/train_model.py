import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv('data/restaurant_sales_data.csv')

df['date'] = pd.to_datetime(df['date'])

# Time Features
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Lag Features
df = df.sort_values('date')

df['lag_1'] = df['quantity_sold'].shift(1)
df['lag_7'] = df['quantity_sold'].shift(7)

# Rolling Average
df['rolling_mean_7'] = df['quantity_sold'].rolling(window=7).mean()

# Remove Missing Values
df = df.dropna()

# Features and Target
X = df[['day_of_week',
        'is_weekend',
        'lag_1',
        'lag_7',
        'rolling_mean_7']]

y = df['quantity_sold']

# Time-Series Split
train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = root_mean_squared_error(y_test, predictions)

print("Improved Model Results")

print("MAE:", mae)

print("RMSE:", rmse)

plt.figure(figsize=(10,5))

plt.plot(y_test.values[:100], label='Actual Sales')

plt.plot(predictions[:100], label='Predicted Sales')

plt.title('Actual vs Predicted Sales')

plt.xlabel('Samples')

plt.ylabel('Quantity Sold')

plt.legend()

plt.show()