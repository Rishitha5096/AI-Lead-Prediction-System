
import joblib
import pandas as pd
from pathlib import Path

# 1. Find project folder
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "ml_model" / "lead_prediction_model.joblib"

# 2. Check whether model exists
if not model_path.exists():
    print("ERROR: Trained model file not found!")
    print("Please run train_model.py first.")
    raise SystemExit

# 3. Load trained model
try:
    model = joblib.load(model_path)
except Exception as error:
    print("ERROR: Unable to load the trained model.")
    print(error)
    raise SystemExit

# 4. Allowed values
valid_sources = [
    "Website", "Email", "Referral", "Cold Call", "LinkedIn"
]

valid_industries = [
    "IT", "Healthcare", "Finance",
    "Retail", "Education", "Manufacturing"
]

# 5. Get valid text input
def get_text_input(message, valid_values):

    while True:
        value = input(message).strip()

        if value.lower() in [item.lower() for item in valid_values]:
            return next(
                item for item in valid_values
                if item.lower() == value.lower()
            )

        print("Invalid input! Choose:", ", ".join(valid_values))


# 6. Get valid numeric input
def get_number_input(message, number_type):

    while True:
        try:
            value = number_type(input(message))

            if value < 0:
                print("Please enter a non-negative value.")
                continue

            return value

        except ValueError:
            print("Invalid number! Please try again.")


# 7. Get new lead details
print("\nENTER NEW CRM LEAD DETAILS")
print("--------------------------")

source = get_text_input(
    "Lead Source (Website/Email/Referral/Cold Call/LinkedIn): ",
    valid_sources
)

industry = get_text_input(
    "Industry (IT/Healthcare/Finance/Retail/Education/Manufacturing): ",
    valid_industries
)

employee_count = get_number_input("Employee Count: ", int)

expected_revenue = get_number_input("Expected Revenue: ", float)

interactions = get_number_input("Number of Interactions: ", int)

lead_age_days = get_number_input("Lead Age in Days: ", int)

# 8. Create new lead DataFrame
new_lead = pd.DataFrame([{
    "source": source,
    "industry": industry,
    "employee_count": employee_count,
    "expected_revenue": expected_revenue,
    "interactions": interactions,
    "lead_age_days": lead_age_days
}])

# 9. Predict outcome
try:
    prediction = model.predict(new_lead)

    probabilities = model.predict_proba(new_lead)[0]

except Exception as error:
    print("ERROR: Prediction failed.")
    print(error)
    raise SystemExit

# 10. Display prediction results
print("\nNEW CRM LEAD PREDICTION")
print("-----------------------")

print("Predicted Outcome:", prediction[0])

for outcome, probability in zip(model.classes_, probabilities):
    print(f"{outcome} Probability: {probability * 100:.2f}%")

# 11. Get Won probability
won_probability = probabilities[
    list(model.classes_).index("Won")
]

# 12. Classify lead
if won_probability >= 0.70:
    category = "Hot"

elif won_probability >= 0.40:
    category = "Warm"

else:
    category = "Cold"

print("Lead Category:", category)
