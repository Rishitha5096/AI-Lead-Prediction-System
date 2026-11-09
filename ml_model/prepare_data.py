import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "crm_leads.csv"

data = pd.read_csv(file_path)

# Separate input features and target
X = data.drop(columns=["lead_id", "outcome"])
y = data["outcome"]

# Convert text columns into numerical values
X = pd.get_dummies(X, columns=["source", "industry"], dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("INPUT FEATURES")
print(X.columns.tolist())

print("\nTRAINING DATA SIZE")
print(X_train.shape)

print("\nTESTING DATA SIZE")
print(X_test.shape)

print("\nTRAINING TARGET")
print(y_train.value_counts())

print("\nTESTING TARGET")
print(y_test.value_counts())