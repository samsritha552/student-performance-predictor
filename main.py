
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Select features and target
features = ["studytime", "failures", "absences", "G1", "G2"]
X = data[features]
y = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("STUDENT PERFORMANCE PREDICTOR")
print("-----------------------------")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# Plot actual vs predicted grades
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.plot([0, 20], [0, 20], linestyle="--")
plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Final Grades")
plt.title("Actual vs Predicted Student Grades")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nGraph saved as actual_vs_predicted.png")

# Get student details
print("\nEnter student details:")

studytime = int(input("Study time (1-4): "))
failures = int(input("Past class failures (0-3): "))
absences = int(input("Number of absences: "))
G1 = float(input("First-period grade (0-20): "))
G2 = float(input("Second-period grade (0-20): "))

# Predict final grade
student = pd.DataFrame(
    [[studytime, failures, absences, G1, G2]],
    columns=features
)

predicted_grade = model.predict(student)[0]

# Grades range from 0 to 20 in this dataset
predicted_grade = max(0, min(20, predicted_grade))

print("\nPredicted final grade:", round(predicted_grade, 2), "/ 20")
