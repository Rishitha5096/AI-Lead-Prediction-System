
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Find and read the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "crm_leads.csv"

data = pd.read_csv(file_path)

# 2. Separate input features and target
X = data.drop(columns=["lead_id", "outcome"])
y = data["outcome"]

# 3. Identify numerical and categorical columns
numeric_columns = [
    "employee_count",
    "expected_revenue",
    "interactions",
    "lead_age_days"
]

categorical_columns = ["source", "industry"]

# 4. Prepare the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# 5. Create the ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
    max_iter=1000,
    random_state=42
))
    ]
)

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict test outcomes
predictions = model.predict(X_test)

# 9. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("MODEL TRAINING COMPLETED")
print("\nActual Outcomes:")
print(y_test.tolist())

print("\nPredicted Outcomes:")
print(predictions.tolist())

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Save the trained model
model_path = project_folder / "ml_model" / "lead_prediction_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(model_path)
