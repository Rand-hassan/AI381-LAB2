
"""Create a deliberately weak model run to test the promotion gate."""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import schema

# Connect to the same MLflow server
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)
mlflow.set_experiment("churn")

# Load and validate data
df = schema.validate(pd.read_csv("churn_v1.csv"))

X = df[schema.FEATURES]
y = df[schema.TARGET]

# Use the same split as the original training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# A deliberately simple and weak baseline
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# Log this run under a separate test batch
with mlflow.start_run(run_name="deliberately_weak_model") as run:
    mlflow.log_param("model", "dummy_classifier")
    mlflow.log_param("data_sha256", "8685886ae7ce")

    mlflow.set_tags({
        "batch": "gate-test",
        "git_sha": "local",
    })

    mlflow.log_metrics({
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
        "p95_latency_ms": 1.0,
    })

    info = mlflow.sklearn.log_model(
        model,
        name="model",
    )

    mlflow.set_tag("model_uri", info.model_uri)

    print("Weak model run created.")
    print("Run ID:", run.info.run_id)
    print("F1:", f1_score(y_test, pred, zero_division=0))
