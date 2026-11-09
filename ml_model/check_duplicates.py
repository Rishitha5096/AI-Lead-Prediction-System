
import pandas as pd
from pathlib import Path

# Find project folder
project_folder = Path(__file__).resolve().parent.parent

# Dataset location
file_path = project_folder / "dataset" / "crm_leads.csv"

# Read dataset
data = pd.read_csv(file_path)

# Check duplicate lead IDs
duplicate_ids = data["lead_id"].duplicated().sum()

# Check completely duplicated rows
duplicate_rows = data.duplicated().sum()

print("DUPLICATE DATA CHECK")
print("--------------------")

print("Total records:", len(data))

print("Duplicate lead IDs:", duplicate_ids)

print("Completely duplicated rows:", duplicate_rows)

# Check whether lead IDs are unique
if duplicate_ids == 0 and duplicate_rows == 0:
    print("\nNo duplicate records found!")
else:
    print("\nDuplicate records detected!")
