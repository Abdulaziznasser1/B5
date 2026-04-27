# ============================================================
#  Employee Attrition Predictor — Training Script
#  Uses scikit-learn (works with Python 3.12)
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Load the dataset ──────────────────────────────────────────
print("Loading Employee Attrition dataset...")

# Try multiple sources
df = None
urls = [
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
    "https://raw.githubusercontent.com/ybifoundation/Dataset/main/HR%20Employee%20Attrition.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/HR-Employee-Attrition.csv",
]

for url in urls:
    try:
        df = pd.read_csv(url)
        if "Attrition" in df.columns and len(df) > 100:
            print(f"Dataset loaded: {df.shape}")
            break
    except:
        continue

if df is None or len(df) < 50:
    raise Exception("Could not load dataset. Check your internet connection.")

print(f"Dataset shape: {df.shape}")
print(f"\nAttrition distribution:\n{df['Attrition'].value_counts()}")

# ── Preprocess ────────────────────────────────────────────────
print("\nPreprocessing data...")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
drop_cols = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
df = df.drop(drop_cols, axis=1)
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ── Split ─────────────────────────────────────────────────────
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")

# ── Train ─────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
print("\n── Model Evaluation ──")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
f1  = f1_score(y_test, y_pred)
print(f"AUC Score : {auc:.3f}  (target: >= 0.80)")
print(f"F1 Score  : {f1:.3f}  (target: >= 0.60)")
print(f"\nDetailed Report:\n{classification_report(y_test, y_pred, target_names=['Stay','Leave'])}")

# ── Top features ──────────────────────────────────────────────
print("── Top 5 Most Important Features ──")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print(importance.to_string())

# ── Save ──────────────────────────────────────────────────────
joblib.dump({"model": model, "feature_names": list(X.columns)}, "attrition_model.pkl")
print("\nModel saved as 'attrition_model.pkl'")

# ── Test prediction ───────────────────────────────────────────
print("\n── Test Prediction ──")
sample = X_test.iloc[[0]]
pred = model.predict(sample)[0]
conf = model.predict_proba(sample)[0][pred]
print(f"Prediction : {'Leave' if pred == 1 else 'Stay'}")
print(f"Confidence : {conf:.1%}")
print(f"Risk Level : {'HIGH' if pred == 1 else 'LOW'}")
print("\nDone! Now run: python serve.py")
