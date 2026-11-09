
import pandas as pd
import random
from pathlib import Path

# 1. Set project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Set dataset output path
output_path = project_folder / "dataset" / "crm_leads.csv"

# 3. Define possible lead sources
sources = [
    "Website",
    "Cold Call",
    "Referral",
    "Email",
    "LinkedIn"
]

# 4. Define industries
industries = [
    "IT",
    "Manufacturing",
    "Healthcare",
    "Education",
    "Retail",
    "Finance"
]

# 5. Generate reproducible sample data
random.seed(42)

leads = []

for lead_id in range(1, 501):

    source = random.choice(sources)
    industry = random.choice(industries)

    employee_count = random.randint(10, 1000)
    expected_revenue = random.randint(10000, 500000)
    interactions = random.randint(1, 15)
    lead_age_days = random.randint(1, 90)

    # 6. Create a simple synthetic outcome rule
    score = (
        interactions * 3
        + (100 - min(lead_age_days, 100)) * 0.15
        + (employee_count / 100)
        + (expected_revenue / 50000)
    )

    if source == "Referral":
        score += 8

    if source == "Website":
        score += 4

    if score >= 35:
        outcome = "Won"
    else:
        outcome = "Lost"

    leads.append({
        "lead_id": lead_id,
        "source": source,
        "industry": industry,
        "employee_count": employee_count,
        "expected_revenue": expected_revenue,
        "interactions": interactions,
        "lead_age_days": lead_age_days,
        "outcome": outcome
    })

# 7. Convert records into a DataFrame
data = pd.DataFrame(leads)

# 8. Save the dataset
data.to_csv(output_path, index=False)

print("NEW DATASET CREATED SUCCESSFULLY!")
print("Total records:", len(data))
print("\nOutcome distribution:")
print(data["outcome"].value_counts())
print("\nDataset saved at:")
print(output_path)
