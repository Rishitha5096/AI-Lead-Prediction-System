# AI-Based Lead Prediction System

## Project Overview

The AI-Based Lead Prediction System is a Machine Learning project that predicts whether a CRM lead is likely to be converted into a customer (Won or Lost).

The system analyzes lead information such as lead source, industry, employee count, expected revenue, number of interactions, and lead age.

It also calculates lead conversion probabilities and classifies leads into Hot, Warm, and Cold categories.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub

## Machine Learning Algorithms

Two Machine Learning algorithms were implemented and compared:

1. Logistic Regression
2. Random Forest Classifier

**Final Model:** Logistic Regression was selected for lead prediction because it achieved better performance in the model comparison and cross-validation experiments.

Random Forest is retained for feature importance analysis.

## Dataset

The project uses a synthetic CRM dataset containing 500 lead records.

### Input Features

- Lead Source
- Industry
- Employee Count
- Expected Revenue
- Number of Interactions
- Lead Age in Days

### Target Variable

- Won
- Lost

## Main Features

- CRM lead dataset generation
- Data cleaning and validation
- Duplicate and missing-value checking
- Machine Learning model training
- Logistic Regression and Random Forest comparison
- 5-fold cross-validation
- Feature importance analysis
- Confusion matrix visualization
- Model performance comparison chart
- Individual lead prediction
- Multiple lead prediction using CSV files
- Lead conversion probability calculation
- Hot, Warm, and Cold lead classification

## Model Performance

### Logistic Regression

- Test Accuracy: 97%
- Cross-validation Accuracy: 97.80%
- Cross-validation Precision: 98.41%
- Cross-validation Recall: 98.65%
- Cross-validation F1 Score: 98.52%

### Random Forest

- Test Accuracy: 92%
- Cross-validation Accuracy: 93.80%

*Note: These results are based on a synthetic dataset. Real-world performance needs validation using actual CRM data.*

## Lead Classification

- Hot Lead: Won probability >= 70%
- Warm Lead: Won probability >= 40% and below 70%
- Cold Lead: Won probability below 40%

## Project Structure

```text
AI-Lead-Prediction-System/
│
├── dataset/
│   ├── crm_leads.csv
│   ├── new_leads.csv
│   ├── lead_predictions.csv
│   └── feature_importance.csv
│
├── documentation/
│   ├── confusion_matrix.png
│   ├── feature_importance_chart.png
│   └── model_performance_chart.png
│
├── ml_model/
│   ├── generate_dataset.py
│   ├── check_dataset.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── compare_models.py
│   ├── cross_validation.py
│   ├── predict_lead.py
│   ├── predict_multiple_leads.py
│   ├── feature_importance.py
│   ├── confusion_matrix.py
│   ├── model_performance_chart.py
│   ├── lead_prediction_model.joblib
│   └── random_forest_model.joblib
│
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
py ml_model/generate_dataset.py
```

### 3. Train the Model

```bash
py ml_model/train_model.py
```

### 4. Predict an Individual Lead

```bash
py ml_model/predict_lead.py
```

### 5. Predict Multiple Leads

```bash
py ml_model/predict_multiple_leads.py
```

## Output Files

- `lead_prediction_model.joblib` – Trained Logistic Regression prediction model.
- `random_forest_model.joblib` – Random Forest model used for feature importance.
- `lead_predictions.csv` – Predictions for multiple CRM leads.
- `feature_importance.csv` – Feature importance results.
- `confusion_matrix.png` – Model evaluation visualization.
- `feature_importance_chart.png` – Feature importance visualization.
- `model_performance_chart.png` – Model comparison visualization.

## Future Enhancements

- Integration with real CRM systems.
- Training with real-world CRM datasets.
- Interactive prediction dashboard.
- Advanced lead scoring and analytics.
- Model deployment using Flask or FastAPI.

## Author

Developed as a Machine Learning project focused on CRM lead conversion prediction.