import joblib

model = joblib.load('restaurant_demand_model.pkl')

sample_data = [[5, 1, 320, 300, 310]]

prediction = model.predict(sample_data)

print("Predicted Restaurant Sales:", prediction[0])