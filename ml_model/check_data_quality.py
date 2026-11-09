
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "crm_leads.csv"

data = pd.read_csv(file_path)

allowed_sources = [
    "Website", "Cold Call", "Referral", "Email", "LinkedIn"
]

allowed_industries = [
    "IT", "Manufacturing", "Healthcare",
    "Education", "Retail", "Finance"
]

allowed_outcomes = ["Won", "Lost"]

print("DATA QUALITY CHECK")
print("------------------")

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check negative numeric values
numeric_columns = [
    "employee_count",
    "expected_revenue",
    "interactions",
    "lead_age_days"
]

for column in numeric_columns:
    print(f"\nNegative values in {column}:",
          (data[column] < 0).sum())

# Check invalid categories
print("\nInvalid sources:")
print(data[~data["source"].isin(allowed_sources)]["source"].unique())

print("\nInvalid industries:")
print(data[~data["industry"].isin(allowed_industries)]["industry"].unique())

print("\nInvalid outcomes:")
print(data[~data["outcome"].isin(allowed_outcomes)]["outcome"].unique())

print("\nDATA QUALITY CHECK COMPLETED!")
