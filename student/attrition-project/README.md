# Employee Attrition Predictor

A machine learning model that predicts whether an employee is likely to leave a company, enabling HR teams to proactively retain talent before it's too late.

---

## The Problem

Companies lose an average of **6–9 months of salary** every time an employee leaves (recruiting, onboarding, lost productivity). Today, HR teams rely on gut feeling or exit interviews — both are too late and inaccurate.

## The ML Solution

**Task:** Binary Classification — predict `Attrition = Yes` or `No`

**Dataset:** IBM HR Analytics Employee Attrition (1,470 employees, 35 features)

**Model:** Best model selected automatically via PyCaret's AutoML (typically Random Forest or Gradient Boosting)

---

## ML Problem Framing

### 1. Goal
Predict employee attrition before it happens so HR can intervene with targeted retention strategies (raises, promotions, flexibility).

### 2. Current / Non-ML Solution
HR managers manually review satisfaction surveys and flag "at-risk" employees based on experience. This is slow, inconsistent, and catches only ~30% of cases before resignation.

### 3. Why ML?
| Factor | Details |
|--------|---------|
| **Difference** | ML catches patterns across 35 variables simultaneously — impossible for humans |
| **Cost** | ~2 hours to train; saves thousands per retained employee |
| **Expertise** | 1 data scientist using PyCaret (no deep ML knowledge required) |
| **Maintenance** | Retrain quarterly with fresh HR data |

### 4. Data Assessment (ART Test)
- **Available:** Yes — IBM HR dataset with 1,470 records, 35 features, clean labels
- **Representative:** Yes — covers all departments, job roles, age groups, and salary bands
- **Trustworthy:** Yes — collected from real HR systems with verified attrition outcomes

**Quantity & Quality:** 1,470 rows × 35 columns. No missing values. Slight class imbalance (~16% attrition) handled by PyCaret automatically.

### 5. Feature Engineering & Selection
PyCaret handles encoding of categorical variables (Department, JobRole, etc.) automatically.

**Top 3 most predictive features:**
1. `OverTime` — Employees working overtime are ~3× more likely to leave
2. `MonthlyIncome` — Lower income strongly correlates with attrition
3. `YearsAtCompany` — Employees in their 1st–2nd year are highest risk

### 6. Prediction & Decision Mapping
- **Prediction:** A label (`Yes` / `No`) + confidence score (0–1)
- **Decision rule:** If `Attrition = Yes` AND `confidence > 0.75` → trigger retention workflow (manager meeting, compensation review)

### 7. Evaluation Metrics
- **Primary metric:** AUC (Area Under Curve) — measures overall model discrimination
- **Secondary:** F1-Score — balances precision and recall for the imbalanced dataset

### 8. Success / Failure Criteria
| Outcome | Criteria |
|---------|----------|
| **Success** | AUC ≥ 0.80 and F1 ≥ 0.60 on the test set |
| **Failure** | AUC < 0.70 or False Positive Rate > 25% (too many false alarms for HR) |

---

## Setup and Usage

### Requirements
```
Python 3.8+
pycaret
fastapi
uvicorn
pandas
```

### Install dependencies
```bash
pip install pycaret fastapi uvicorn pandas
```

### Step 1 — Train the model
```bash
python train.py
```
This will:
- Download the dataset automatically
- Train and compare multiple ML models
- Save the best model as `attrition_model.pkl`
- Run a quick test prediction

### Step 2 — Start the API server
```bash
python serve.py
```

### Step 3 — Use the API
Open your browser and go to:
```
http://127.0.0.1:8000/docs
```
This opens an interactive UI where you can send employee data and get predictions.

Or call the API directly:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Age": 28, "OverTime": "Yes", "MonthlyIncome": 3000, "YearsAtCompany": 1, ...}'
```

**Example response:**
```json
{
  "attrition_risk": "Yes",
  "confidence": 0.87,
  "risk_level": "HIGH",
  "recommendation": "Flag for retention program — consider raise, promotion, or flexibility options."
}
```

---

## Project Structure
```
├── train.py          # Train the model
├── serve.py          # Run the prediction API
├── attrition_model.pkl  # Saved model (generated after training)
└── README.md
```