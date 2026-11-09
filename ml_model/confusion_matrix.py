
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Load dataset
data_path = project_folder / "dataset" / "crm_leads.csv"
data = pd.read_csv(data_path)

# 3. Separate input features and target
X = data.drop(columns=["lead_id", "outcome"])
y = data["outcome"]

# 4. Split dataset (same settings used during training)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Load trained model
model_path = project_folder / "ml_model" / "lead_prediction_model.joblib"
model = joblib.load(model_path)

# 6. Predict test data
predictions = model.predict(X_test)

# 7. Create confusion matrix
labels = ["Lost", "Won"]

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

# 8. Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")

plt.title("AI Lead Prediction - Logistic Regression Confusion Matrix")
plt.tight_layout()

# 9. Save chart
output_file = (
    project_folder / "documentation" / "confusion_matrix.png"
)

plt.savefig(output_file, dpi=300)

# 10. Show results
plt.show()

print("\nCONFUSION MATRIX")
print(matrix)

print("\nConfusion matrix saved successfully!")
print(output_file)
