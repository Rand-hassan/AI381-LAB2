from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "churn_v1.csv"
MODEL_PATH = PROJECT_ROOT / "model.joblib"

df = pd.read_csv(DATA)

numeric = ["tenure_months", "monthly_charges", "support_calls"]
categorical = ["contract", "internet_service"]

X = df[numeric + categorical]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]

print("F1 :", round(f1_score(y_test, proba > 0.5), 3))
print("AUC:", round(roc_auc_score(y_test, proba), 3))

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
