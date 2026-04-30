import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students_1000_advanced.csv")

print("Dataset Loaded ✅")
print(df.head())

# Features & Target
X = df[['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks']]
y = df['Final_Result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualization
sns.scatterplot(x=df['Study_Hours'], y=df['Previous_Marks'], hue=df['Final_Result'])
plt.title("Study Hours vs Marks")
plt.show()

# Test Prediction
sample = [[6, 85, 70, 75, 78]]
prediction = model.predict(sample)

print("\nSample Prediction:", "Pass" if prediction[0] == 1 else "Fail")