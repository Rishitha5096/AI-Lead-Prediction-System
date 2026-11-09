
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Read feature importance data
file_path = project_folder / "dataset" / "feature_importance.csv"
data = pd.read_csv(file_path)

# 3. Remove preprocessing prefixes
data["Feature"] = data["Feature"].str.replace(
    "numeric__", "", regex=False
).str.replace(
    "categorical__", "", regex=False
)

# 4. Select top 10 features
data = data.head(10).sort_values(
    by="Importance",
    ascending=True
)

# 5. Create bar chart
plt.figure(figsize=(10, 6))

plt.barh(
    data["Feature"],
    data["Importance"] * 100
)

plt.xlabel("Importance (%)")
plt.ylabel("CRM Features")
plt.title("AI Lead Prediction - Feature Importance")

plt.tight_layout()

# 6. Save chart
output_file = project_folder / "documentation" / "feature_importance_chart.png"

plt.savefig(output_file, dpi=300)

# 7. Display chart
plt.show()

print("Chart saved successfully!")
print(output_file)

