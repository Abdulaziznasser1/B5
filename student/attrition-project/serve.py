# ============================================================
#  Employee Attrition Predictor — API Server
#  Run: python serve.py
#  Then open: http://127.0.0.1:8000/docs
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# ── Load the trained model ────────────────────────────────────
print("Loading model...")
data = joblib.load("attrition_model.pkl")
model = data["model"]
feature_names = data["feature_names"]
print("Model loaded. Starting API...")

app = FastAPI(
    title="Employee Attrition Predictor",
    description="Predicts whether an employee is likely to leave the company.",
    version="1.0"
)

# ── Define the input schema ───────────────────────────────────
class Employee(BaseModel):
    Age: int = 35
    BusinessTravel: int = 1      # 0=Non-Travel, 1=Travel_Rarely, 2=Travel_Frequently
    DailyRate: int = 800
    Department: int = 1          # 0=HR, 1=R&D, 2=Sales
    DistanceFromHome: int = 10
    Education: int = 3           # 1-5
    EducationField: int = 2      # 0-5
    EnvironmentSatisfaction: int = 3  # 1-4
    Gender: int = 1              # 0=Female, 1=Male
    HourlyRate: int = 65
    JobInvolvement: int = 3      # 1-4
    JobLevel: int = 2            # 1-5
    JobRole: int = 6             # 0-8
    JobSatisfaction: int = 3     # 1-4
    MaritalStatus: int = 1       # 0=Divorced, 1=Married, 2=Single
    MonthlyIncome: int = 5000
    MonthlyRate: int = 15000
    NumCompaniesWorked: int = 2
    OverTime: int = 0            # 0=No, 1=Yes
    PercentSalaryHike: int = 15
    PerformanceRating: int = 3   # 3=Excellent, 4=Outstanding
    RelationshipSatisfaction: int = 3  # 1-4
    StockOptionLevel: int = 1    # 0-3
    TotalWorkingYears: int = 10
    TrainingTimesLastYear: int = 3
    WorkLifeBalance: int = 3     # 1-4
    YearsAtCompany: int = 5
    YearsInCurrentRole: int = 3
    YearsSinceLastPromotion: int = 1
    YearsWithCurrManager: int = 3

# ── API Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Employee Attrition API is running! Go to /docs for the interactive UI."}

@app.post("/predict")
def predict(employee: Employee):
    """
    Send employee data and get attrition prediction.
    Returns: Leave/Stay prediction + confidence score.
    """
    input_df = pd.DataFrame([employee.dict()])[feature_names]

    pred = model.predict(input_df)[0]
    conf = float(model.predict_proba(input_df)[0][pred])

    label = "Yes" if pred == 1 else "No"

    return {
        "attrition_risk": label,
        "confidence": round(conf, 3),
        "risk_level": "HIGH" if pred == 1 else "LOW",
        "recommendation": (
            "Flag for retention program — consider raise, promotion, or flexibility."
            if pred == 1
            else "Employee appears stable. Continue regular engagement."
        )
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Run the server ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)