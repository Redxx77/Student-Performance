import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset

df = pd.read_csv("students_1000_advanced.csv")

features = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks']

X = df[features].values   # numpy
y = df['Final_Result']

# Train model

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model

joblib.dump(model, "model.pkl")

print("✅ model.pkl saved (Pass/Fail only)")
