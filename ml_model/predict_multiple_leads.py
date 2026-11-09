
import joblib
import pandas as pd
from pathlib import Path

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Define file paths
model_path = project_folder / "ml_model" / "lead_prediction_model.joblib"
input_file = project_folder / "dataset" / "new_leads.csv"
output_file = project_folder / "dataset" / "lead_predictions.csv"

# 3. Check model file
if not model_path.exists():
    print("ERROR: Trained model not found!")
    raise SystemExit

# 4. Check input CSV
if not input_file.exists():
    print("ERROR: new_leads.csv not found!")
    raise SystemExit

# 5. Load model
try:
    model = joblib.load(model_path)
except Exception as error:
    print("ERROR: Unable to load model.")
    print(error)
    raise SystemExit

# 6. Read CSV
try:
    leads = pd.read_csv(input_file)
except pd.errors.EmptyDataError:
    print("ERROR: CSV file is empty!")
    raise SystemExit
except Exception as error:
    print("ERROR: Unable to read CSV.")
    print(error)
    raise SystemExit

# 7. Check empty dataset
if leads.empty:
    print("ERROR: No lead records found!")
    raise SystemExit

# 8. Required columns
required_columns = [
    "source", "industry", "employee_count",
    "expected_revenue", "interactions", "lead_age_days"
]

# 9. Check missing columns
missing_columns = [
    col for col in required_columns if col not in leads.columns
]

if missing_columns:
    print("ERROR: Missing columns:", missing_columns)
    raise SystemExit

leads = leads[required_columns].copy()

# 10. Validate source and industry
allowed_sources = [
    "Website", "Cold Call", "Referral", "Email", "LinkedIn"
]

allowed_industries = [
    "IT", "Manufacturing", "Healthcare",
    "Education", "Retail", "Finance"
]

for column, allowed_values in [
    ("source", allowed_sources),
    ("industry", allowed_industries)
]:
    leads[column] = leads[column].astype("string").str.strip()

    invalid_values = (
        leads[column].isna()
        | ~leads[column].isin(allowed_values)
    )

    if invalid_values.any():
        print(f"ERROR: Invalid values found in {column}:")
        print(leads.loc[invalid_values, column].tolist())
        print("Allowed values:", allowed_values)
        raise SystemExit

# 11. Validate numerical columns
numeric_columns = [
    "employee_count", "expected_revenue",
    "interactions", "lead_age_days"
]

for column in numeric_columns:
    leads[column] = pd.to_numeric(
        leads[column], errors="coerce"
    )

    if leads[column].isna().any():
        print(f"ERROR: Missing or invalid numbers in {column}!")
        raise SystemExit

    if (leads[column] < 0).any():
        print(f"ERROR: Negative values in {column}!")
        raise SystemExit

# 12. Predict outcomes
try:
    predictions = model.predict(leads)
    probabilities = model.predict_proba(leads)
except Exception as error:
    print("ERROR: Prediction failed.")
    print(error)
    raise SystemExit

# 13. Find Won probability
won_index = list(model.classes_).index("Won")
won_probabilities = probabilities[:, won_index]

# 14. Classify leads
categories = []

for probability in won_probabilities:
    if probability >= 0.70:
        categories.append("Hot")
    elif probability >= 0.40:
        categories.append("Warm")
    else:
        categories.append("Cold")

# 15. Add prediction results
leads["predicted_outcome"] = predictions
leads["won_probability"] = won_probabilities * 100
leads["lead_category"] = categories

# 16. Save predictions
leads.to_csv(output_file, index=False)

# 17. Display results
print("\nMULTIPLE LEAD PREDICTIONS")
print("-------------------------")
print(leads.to_string(index=False))

print("\nPredictions saved successfully!")
print(output_file)
