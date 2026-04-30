import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Student Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")

# Load dataset

df = pd.read_csv("students_1000_advanced.csv")

# Load model

model = joblib.load("model.pkl")

# Inputs

study_hours = st.slider("Study Hours", 0.0, 10.0, 5.0)
attendance = st.slider("Attendance", 0, 100, 75)
previous_marks = st.slider("Previous Marks", 0, 100, 60)
assignments = st.slider("Assignments", 0, 100, 60)
internal_marks = st.slider("Internal Marks", 0, 100, 60)

# Load dataset
df = pd.read_csv("students_1000_advanced.csv")

# ✅ ADD THIS
features = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks']

# Button

if st.button("Predict"):

    input_data = np.array([[ 
        float(study_hours),
        float(attendance),
        float(previous_marks),
        float(assignments),
        float(internal_marks)
    ]], dtype=np.float64)

    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if result == 1:
        st.success(f"PASS ✅ ({prob:.2f})")
    else:
        st.error(f"FAIL ❌ ({prob:.2f})")

st.subheader("📊 Study Hours vs Previous Marks")



fig, ax = plt.subplots()

sns.scatterplot(
    x=df['Study_Hours'],
    y=df['Previous_Marks'],
    hue=df['Final_Result'],
    ax=ax
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Previous Marks")

st.pyplot(fig)

st.subheader("📉 Confusion Matrix")

features = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks']

# ✅ ADD THIS LINE
X = df[features].values
y = df['Final_Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)