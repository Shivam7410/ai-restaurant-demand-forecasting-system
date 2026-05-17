import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv('data/restaurant_sales_data.csv')

df['date'] = pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek

df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

X = df[['day_of_week', 'is_weekend']]

y = df['quantity_sold']

train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = root_mean_squared_error(y_test, predictions)

print("MAE:", mae)

print("RMSE:", rmse)