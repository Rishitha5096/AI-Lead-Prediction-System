import pandas as pd
from pathlib import Path

# Find the CSV file
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "crm_leads.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Display the dataset
print("CRM LEAD DATASET")
print(data)

# Display the first 5 rows
print("\nFIRST 5 LEADS")
print(data.head())

# Display dataset information
print("\nDATASET INFORMATION")
print(data.info())

# Check for missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Count Won and Lost leads
print("\nLEAD OUTCOMES")
print(data["outcome"].value_counts())