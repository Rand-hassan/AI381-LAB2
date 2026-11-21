import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = os.getenv("DATA_FILE", str(PROJECT_ROOT / "churn_v1.csv"))

EXPERIMENT = "churn"

SLICE_COLUMN = "contract"

BATCH = os.getenv("BATCH", os.getenv("GIT_SHA", "local")[:7])

def preprocessing() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), schema.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), schema.CATEGORICAL),
    ])

CANDIDATES = {
    "logreg": LogisticRegression(
        max_iter=1000,
        C=0.01,
    ),
}

def p95_latency_ms(model, X: pd.DataFrame, n: int = 200) -> float:
    times = []

    for i in range(n):
        row = X.iloc[[i % len(X)]]

        start = time.perf_counter()
        model.predict_proba(row)
        elapsed = (time.perf_counter() - start) * 1000

        times.append(elapsed)

    return float(np.percentile(times, 95))


def size_kb(model) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.joblib"

        joblib.dump(model, path)

        return path.stat().st_size / 1024

   
def main():
    # Connect to MLflow
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    mlflow.set_experiment(EXPERIMENT)

    # Load and validate dataset
    df = schema.validate(pd.read_csv(DATA_FILE))

    X = df[schema.FEATURES]
    y = df[schema.TARGET]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Calculate dataset hash
    data_sha = hashlib.sha256(
        Path(DATA_FILE).read_bytes()
    ).hexdigest()[:12]

    # Train all candidate models
    for name, clf in CANDIDATES.items():

        model = Pipeline([
            ("prep", preprocessing()),
            ("clf", clf),
        ])

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = proba > 0.5

        with mlflow.start_run(run_name=name) as run:

            # Log parameters
            mlflow.log_param("model", name)
            mlflow.log_params({
                f"clf__{k}": v
                for k, v in clf.get_params().items()
            })
            mlflow.log_param("data_file", DATA_FILE)
            mlflow.log_param("data_sha256", data_sha)

            # Log tags
            mlflow.set_tags({
                "batch": BATCH,
                "git_sha": os.getenv("GIT_SHA", "local"),
            })

            # Calculate performance metrics
            metrics = {
                "auc": roc_auc_score(y_test, proba),
                "f1": f1_score(y_test, pred),
                "p95_latency_ms": p95_latency_ms(model, X_test),
                "size_kb": size_kb(model),
            }

            # Calculate F1 for each contract type
            for value in X_test[SLICE_COLUMN].unique():
                mask = (X_test[SLICE_COLUMN] == value).to_numpy()

                metrics[f"f1_{SLICE_COLUMN}_{value}"] = f1_score(
                    y_test[mask],
                    pred[mask],
                    zero_division=0,
                )

            # Log metrics
            mlflow.log_metrics({
                k: round(float(v), 4)
                for k, v in metrics.items()
            })

            # Log model with signature
            signature = infer_signature(X_test, proba)

            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_test.head(3),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            mlflow.set_tag("model_uri", info.model_uri)

            print(f"\nModel: {name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Metrics: {metrics}")

    # Print experiment information
    print(json.dumps({
        "batch": BATCH,
        "experiment": EXPERIMENT,
        "data_sha256": data_sha,
    }, indent=2))


if __name__ == "__main__":
    main()



