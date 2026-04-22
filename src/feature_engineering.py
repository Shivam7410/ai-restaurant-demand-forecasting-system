import pandas as pd

df = pd.read_csv('data/restaurant_sales_data.csv')

df['date'] = pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek

df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# print(df[['date', 'day_of_week', 'is_weekend']].head())

df = df.sort_values('date')

df['lag_1'] = df['quantity_sold'].shift(1)

df['lag_7'] = df['quantity_sold'].shift(7)

# print(df[['date', 'quantity_sold', 'lag_1', 'lag_7']].head(10))

df['rolling_mean_7'] = df['quantity_sold'].rolling(window=7).mean()

# print(df[['quantity_sold', 'rolling_mean_7']].head(10))

df = df.dropna()

# print(df.shape)

# print(df.head())

train_size = int(len(df)*0.8)

train = df.iloc[:train_size]

test = df.iloc[train_size:]

# print("Train Shape:", train.shape)

# print("Test Shape:", test.shape)

categorical_cols = [
    'restaurant_type',
    'menu_item_name',
    'meal_type',
    'weather_condition'
]

df = pd.get_dummies(df, columns=categorical_cols)

# print(df.head())

y = df['quantity_sold']

X = df.drop(['quantity_sold', 'date', 'key_ingredients_tags'], axis=1)

print("Feature Shape:", X.shape)

print("Target Shape:", y.shape)