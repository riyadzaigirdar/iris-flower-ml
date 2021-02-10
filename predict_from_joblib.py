import joblib

model = joblib.load("trained_model.joblib")

print(model.predict([[6, 2, 1.6, 0.2]]))
