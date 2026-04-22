import pandas as pd

df = pd.read_csv('data/restaurant_sales_data.csv')

df['date'] = pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek

df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

print(df[['date', 'day_of_week', 'is_weekend']].head())