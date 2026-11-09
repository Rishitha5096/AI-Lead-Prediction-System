
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Store latest cross-validation results
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Logistic Regression": [97.80, 98.41, 98.65, 98.52],
    "Random Forest": [93.80, 94.54, 97.30, 95.89]
}

# 3. Create DataFrame
df = pd.DataFrame(data)

# 4. Set metric names as index
df = df.set_index("Metric")

# 5. Create comparison chart
ax = df.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.7
)

# 6. Add chart labels
plt.title("AI Lead Prediction - Model Performance Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 110)
plt.xticks(rotation=0)
plt.legend(title="Machine Learning Models")

# 7. Add values above bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()

# 8. Save chart
output_file = (
    project_folder / "documentation" / "model_performance_chart.png"
)

plt.savefig(output_file, dpi=300)

# 9. Display chart
plt.show()

print("\nMODEL PERFORMANCE")
print("-----------------")
print(df)

print("\nPerformance chart saved successfully!")
print(output_file)
