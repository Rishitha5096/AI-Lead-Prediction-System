
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Read dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "crm_leads.csv"

data = pd.read_csv(file_path)

# 2. Separate features and target
X = data.drop(columns=["lead_id", "outcome"])
y = data["outcome"]

# 3. Identify columns
numeric_columns = [
    "employee_count",
    "expected_revenue",
    "interactions",
    "lead_age_days"
]

categorical_columns = ["source", "industry"]

# 4. Create cross-validation strategy
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 5. Define algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 6. Compare models using cross-validation
for name, algorithm in algorithms.items():

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", algorithm)
        ]
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cross_validation,
        scoring={
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label="Won",
        zero_division=0
    ),
    "recall": make_scorer(
        recall_score,
        pos_label="Won",
        zero_division=0
    ),
    "f1": make_scorer(
        f1_score,
        pos_label="Won",
        zero_division=0
    )
}
    )

    print("\nMODEL:", name)
    print("-----------------------------")

    print(
        f"Average Accuracy: {scores['test_accuracy'].mean() * 100:.2f}%"
    )

    print(
        f"Average Precision: {scores['test_precision'].mean() * 100:.2f}%"
    )

    print(
        f"Average Recall: {scores['test_recall'].mean() * 100:.2f}%"
    )

    print(
        f"Average F1 Score: {scores['test_f1'].mean() * 100:.2f}%"
    )
