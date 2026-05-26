import streamlit as st
import pandas as pd
import joblib

model = joblib.load('restaurant_demand_model.pkl')

st.title("Restaurant Demand Forecasting System")

day_of_week = st.number_input(
    "Day of Week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6
)

is_weekend = st.number_input(
    "Is Weekend? (0=No, 1=Yes)",
    min_value=0,
    max_value=1
)

lag_1 = st.number_input(
    "Previous Day Sales"
)

lag_7 = st.number_input(
    "Previous Week Sales"
)

rolling_mean_7 = st.number_input(
    "7-Day Average Sales"
)

if st.button("Predict Sales"):
        input_data = pd.DataFrame(
        [[
            day_of_week,
            is_weekend,
            lag_1,
            lag_7,
            rolling_mean_7
        ]],
        columns=[
            'day_of_week',
            'is_weekend',
            'lag_1',
            'lag_7',
            'rolling_mean_7'
        ]
    )
        prediction = model.predict(input_data)
        st.success(
        f"Predicted Restaurant Sales: {prediction[0]:.2f}"
        )