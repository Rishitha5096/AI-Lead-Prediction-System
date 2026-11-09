
import joblib
import pandas as pd
from pathlib import Path

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Load trained model
model_path = project_folder / "ml_model" / "random_forest_model.joblib"
model = joblib.load(model_path)

# 3. Get preprocessing step
preprocessor = model.named_steps["preprocessor"]

# 4. Get Random Forest classifier
classifier = model.named_steps["classifier"]

# 5. Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# 6. Get feature importance values
importance_values = classifier.feature_importances_

# 7. Create a DataFrame
importance_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# 8. Sort features by importance
importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)

# 9. Display results
print("\nFEATURE IMPORTANCE")
print("------------------")

print(importance_data.to_string(index=False))

# 10. Save results
output_file = project_folder / "dataset" / "feature_importance.csv"

importance_data.to_csv(output_file, index=False)

print("\nFeature importance saved successfully!")
print(output_file)
