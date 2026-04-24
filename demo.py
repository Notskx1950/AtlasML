#!/usr/bin/env python3
"""AtlasML demo script — exercises the full API lifecycle."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib

from pathlib import Path

BASE_URL = os.getenv("ATLASML_URL", "http://localhost:8000")


def step(n: int, msg: str) -> None:
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


def main() -> None:
    """Run the full demo."""
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    # --- Prepare artifacts ---
    tmpdir = tempfile.mkdtemp(prefix="atlasml_demo_")

    PROJECT_ROOT = Path(__file__).resolve().parent
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Train model v1
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    clf_v1 = LogisticRegression(C=1.0, random_state=42).fit(X, y)
    host_model_v1_path = ARTIFACTS_DIR / "model_v1.joblib"
    joblib.dump(clf_v1, host_model_v1_path)

    container_model_v1_path = f"/app/artifacts/{host_model_v1_path.name}"

    # Train model v2 (different hyperparams)
    clf_v2 = LogisticRegression(C=10.0, random_state=42).fit(X, y)
    host_model_v2_path = ARTIFACTS_DIR / "model_v2.joblib"
    joblib.dump(clf_v2, host_model_v2_path)

    container_model_v2_path = f"/app/artifacts/{host_model_v2_path.name}"

    # Create test dataset
    dataset_path = ARTIFACTS_DIR / "test_dataset.jsonl"
    X_test, y_test = make_classification(n_samples=100, n_features=4, random_state=99)
    with open(dataset_path, "w") as f:
        for i in range(len(X_test)):
            row = {
                "input": {f"f{j}": float(X_test[i][j]) for j in range(4)},
                "label": int(y_test[i]),
            }
            f.write(json.dumps(row) + "\n")

    container_dataset_path = f"/app/artifacts/{dataset_path.name}"

    # --- Step 1: Register model v1 ---
    step(1, "Register model v1")
    resp = client.post(
        "/models/register",
        json={"name": "demo-classifier", "version": "v1", "artifact_uri": container_model_v1_path},
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2, default=str)}")

    # --- Step 2: Activate v1 ---
    step(2, "Activate v1")
    resp = client.post(
        "/models/demo-classifier/activate", json={"version": "v1"}
    )
    print("Status:", resp.status_code)
    print("Headers:", resp.headers.get("content-type"))
    print("Raw text:", repr(resp.text))
    print(f"Active version: {resp.json()['version']} (is_active={resp.json()['is_active']})")
    # --- Step 3: Sync predict ---
    step(3, "POST /predict with 5 test inputs")
    test_inputs = [
        {f"f{j}": float(X_test[i][j]) for j in range(4)} for i in range(5)
    ]
    resp = client.post(
        "/predict",
        json={"model_name": "demo-classifier", "inputs": test_inputs},
    )
    data = resp.json()
    print(data)
    print(f"Predictions: {[p['prediction'] for p in data['predictions']]}")
    print(f"Model version: {data['model_version']}")
    print(f"Latency: {data['latency_ms']:.2f} ms")

    # --- Step 4: Eval v1 ---
    step(4, "POST /eval/run on 100-row test set for v1")
    resp = client.post(
        "/eval/run",
        json={
            "model_name": "demo-classifier",
            "version": "v1",
            "dataset_id": "demo-test",
            "dataset_path": container_dataset_path,
        },
    )
    run_id_v1 = resp.json()["run_id"]
    print(f"Eval run ID: {run_id_v1}")

    # Poll until complete
    for _ in range(60):
        resp = client.get(f"/eval/runs/{run_id_v1}")
        status = resp.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(1)

    # --- Step 5: Print v1 metrics ---
    step(5, "Print eval metrics for v1")
    resp = client.get(f"/eval/runs/{run_id_v1}")
    run_data = resp.json()
    print(f"Status: {run_data['status']}")
    if run_data.get("metrics"):
        for m in run_data["metrics"]:
            print(f"  {m['metric_name']}: {m['value']:.4f}")
    else:
        print("  (No metrics — eval may still be running or requires a worker)")

    # --- Step 6: Register model v2 ---
    step(6, "Register model v2")
    resp = client.post(
        "/models/register",
        json={"name": "demo-classifier", "version": "v2", "artifact_uri": container_model_v2_path},
    )
    print(f"Registered v2: {resp.json()['version']}")

    # --- Step 7: Eval v2 ---
    step(7, "POST /eval/run for v2")
    resp = client.post(
        "/eval/run",
        json={
            "model_name": "demo-classifier",
            "version": "v2",
            "dataset_id": "demo-test",
            "dataset_path": container_dataset_path,
        },
    )
    run_id_v2 = resp.json()["run_id"]
    print(f"Eval run ID: {run_id_v2}")

    for _ in range(60):
        resp = client.get(f"/eval/runs/{run_id_v2}")
        if resp.json()["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    # --- Step 8: Compare v1 vs v2 ---
    step(8, "GET /eval/compare v1 vs v2")
    resp = client.get(
        "/eval/compare",
        params={"model_name": "demo-classifier", "v1": "v1", "v2": "v2"},
    )
    if resp.status_code == 200:
        comparison = resp.json()
        print(f"Model: {comparison['model_name']}")
        print(f"{'Metric':<25} {'v1':>10} {'v2':>10} {'Delta':>10} {'%Change':>10}")
        print("-" * 65)
        for c in comparison["comparison"]:
            print(
                f"{c['metric']:<25} {c['v1']:>10.4f} {c['v2']:>10.4f} "
                f"{c['delta']:>+10.4f} {c['pct_change']:>+9.2f}%"
            )
    else:
        print(f"  Compare not available (requires completed eval runs): {resp.text}")

    # --- Step 9: Activate v2 ---
    step(9, "Activate v2 (traffic switch)")
    resp = client.post(
        "/models/demo-classifier/activate", json={"version": "v2"}
    )
    print(f"Active version: {resp.json()['version']}")

    # Verify predictions use v2
    resp = client.post(
        "/predict",
        json={"model_name": "demo-classifier", "inputs": test_inputs[:2]},
    )
    print(f"Predict now uses version: {resp.json()['model_version']}")

    # --- Step 10: Simulate bad model ---
    step(10, "Simulate broken model v3")
    resp = client.post(
        "/models/register",
        json={
            "name": "demo-classifier",
            "version": "v3",
            "artifact_uri": "/nonexistent/bad_model.joblib",
        },
    )
    client.post("/models/demo-classifier/activate", json={"version": "v3"})
    print("Activated v3 (broken artifact)")

    resp = client.post(
        "/predict",
        json={"model_name": "demo-classifier", "inputs": test_inputs[:1]},
    )
    print(f"Predict status: {resp.status_code}")
    print(f"Error: {resp.json().get('detail', resp.json())}")

    # --- Step 11: Rollback to v1 ---
    step(11, "Rollback to v1")
    resp = client.post(
        "/models/demo-classifier/activate", json={"version": "v1"}
    )
    print(f"Rolled back to: {resp.json()['version']}")

    resp = client.post(
        "/predict",
        json={"model_name": "demo-classifier", "inputs": test_inputs[:2]},
    )
    print(f"Predictions resumed: {[p['prediction'] for p in resp.json()['predictions']]}")
    print(f"Using version: {resp.json()['model_version']}")

    print(f"\n{'='*60}")
    print("  Demo complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
