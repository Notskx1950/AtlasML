# AtlasML Debugging Notes

This document records the main issues encountered while bringing up the AtlasML demo in a Dockerized local environment.

The goal of this debugging process was to make the end-to-end demo run successfully through the full model lifecycle:

1. register model versions
2. activate a serving version
3. run synchronous prediction
4. run offline evaluation
5. compare model versions
6. switch active versions
7. simulate a broken artifact
8. roll back to a stable version

The issues covered several layers of the system, including Alembic migrations, Docker networking, database constraints, artifact path handling, JSON serialization, runtime dependencies, and API error handling.

---

## 1. Alembic could not import the application package

### Symptom

Running the migration command inside the API container:

```bash
docker compose exec api alembic upgrade head
````

failed with:

```text
ModuleNotFoundError: No module named 'app'
```

### Root Cause

Alembic executes the migration environment file:

```text
alembic/env.py
```

which imports the SQLAlchemy metadata from the application:

```python
from app.db.models import Base
```

When Alembic is launched through the installed CLI script, typically located at:

```text
/usr/local/bin/alembic
```

Python may not automatically include the project root directory in `sys.path`.

In this project, the application package lives under the container project root:

```text
/app/app
```

For `from app.db.models import Base` to work, Python needs `/app` in its module search path. Without it, Alembic cannot resolve the `app` package.

### Fix

Ensure the project root is visible to Python inside the container.

One option is to set `PYTHONPATH` in `docker-compose.yml`:

```yaml
environment:
  PYTHONPATH: /app
```

Another option is to insert the project root explicitly in `alembic/env.py`:

```python
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
```

### Engineering Takeaway

Alembic is a Python-based migration tool. When it executes migration scripts, it must be able to import the application's ORM models. In Dockerized environments, the Python import path should be configured explicitly instead of relying on the CLI entrypoint's default behavior.

---

## 2. Alembic tried to connect to `localhost:5432`

### Symptom

After resolving the import issue, Alembic failed to connect to Postgres:

```text
connection to server at "localhost", port 5432 failed
```

### Root Cause

Inside a Docker container, `localhost` refers to the current container itself.

The migration was being executed from the `api` container, so:

```text
localhost:5432
```

pointed to the API container, not the Postgres container.

The Postgres service in `docker-compose.yml` was named:

```yaml
postgres:
```

Therefore, container-to-container communication should use:

```text
postgres:5432
```

The original `alembic.ini` contained a local database URL:

```ini
sqlalchemy.url = postgresql+psycopg2://atlas:atlas@localhost:5432/atlasml
```

which was invalid from inside the API container.

### Fix

Update the Alembic database URL to use the Docker Compose service name:

```ini
sqlalchemy.url = postgresql+psycopg2://atlas:atlas@postgres:5432/atlasml
```

A more flexible long-term approach is to let Alembic read a sync database URL from the environment:

```env
DATABASE_SYNC_URL=postgresql+psycopg2://atlas:atlas@postgres:5432/atlasml
```

and override `sqlalchemy.url` in `alembic/env.py`.

### Engineering Takeaway

Docker Compose services should communicate using service names, not `localhost`. `localhost` inside a container means the container itself.

---

## 3. Server-side errors were initially masked as JSON parsing errors

### Symptom

During model activation, the demo client failed at:

```python
resp.json()
```

with:

```text
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

### Root Cause

The client assumed that every response was valid JSON. However, the API was returning:

```text
500 Internal Server Error
```

with a plain text body:

```text
Internal Server Error
```

The JSON parsing error was only a secondary client-side error. The real issue was on the server side.

### Fix

Before parsing the response as JSON, inspect the raw response:

```python
print("Status:", resp.status_code)
print("Headers:", resp.headers.get("content-type"))
print("Raw text:", repr(resp.text))

if resp.status_code >= 400:
    raise RuntimeError(f"Request failed: {resp.status_code} {resp.text}")
```

Then inspect server logs:

```bash
docker compose logs api --tail 200
```

### Engineering Takeaway

Client-side JSON parsing errors can hide server-side failures. During debugging, always inspect the HTTP status code and raw response body before assuming the response matches the success schema.

---

## 4. Duplicate model versions caused activation failure

### Symptom

The activation endpoint returned `500 Internal Server Error`.

The API logs showed:

```text
sqlalchemy.exc.MultipleResultsFound: Multiple rows were found when one or none was required
```

The failure occurred at:

```python
result.scalar_one_or_none()
```

### Root Cause

Repeated runs of `demo.py` inserted duplicate model-version records:

```text
name = demo-classifier
version = v1
```

The activation logic queried by `(name, version)` and expected this pair to uniquely identify a single model version. However, the database allowed multiple identical `(name, version)` rows.

This created a mismatch:

* the registry allowed duplicate model-version records
* the activation logic assumed model versions were unique

### Fix

Add a uniqueness constraint to the SQLAlchemy model:

```python
__table_args__ = (
    UniqueConstraint("name", "version", name="uq_model_name_version"),
)
```

Also ensure the activation query filters by both model name and version:

```python
select(ModelVersion).where(
    ModelVersion.name == name,
    ModelVersion.version == body.version,
)
```

### Engineering Takeaway

Registry invariants should be enforced at the database level. If application logic assumes `(model_name, version)` is unique, the database schema should enforce that invariant directly.

---

## 5. ORM constraints must be synced with Alembic migrations

### Symptom

Adding `UniqueConstraint("name", "version")` to the SQLAlchemy model did not initially prevent duplicate records after rebuilding the database.

### Root Cause

Updating the SQLAlchemy ORM model changes the Python-side table definition, but it does not automatically update the actual Postgres schema.

The database tables were created by the Alembic migration file:

```text
alembic/versions/001_initial_schema.py
```

If the migration file does not include the unique constraint, the real database table will still allow duplicates.

### Fix

Update the Alembic migration so the `model_versions` table includes:

```python
sa.UniqueConstraint("name", "version", name="uq_model_name_version")
```

Example:

```python
op.create_table(
    "model_versions",
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False, index=True),
    sa.Column("version", sa.String(64), nullable=False),
    sa.Column("artifact_uri", sa.String(1024), nullable=False),
    sa.Column("runtime_config", JSONB, nullable=True),
    sa.Column("metadata_", JSONB, nullable=True),
    sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name", "version", name="uq_model_name_version"),
)
```

Then rebuild the database:

```bash
docker compose down -v
docker compose up --build -d
docker compose exec api alembic upgrade head
```

### Validation

After applying the migration change and rebuilding the database, repeated registration of the same `(name, version)` was rejected at registration time instead of causing activation-time failure.

### Engineering Takeaway

The ORM model and database migration files must stay consistent. Database constraints only take effect when they are applied through migrations or direct schema changes.

---

## 6. The demo was not repeatable after adding uniqueness constraints

### Symptom

After enforcing `(name, version)` uniqueness, running `demo.py` a second time failed at Step 1 because it attempted to register the same model version again:

```text
demo-classifier:v1
```

### Root Cause

The demo used fixed identifiers:

```python
"name": "demo-classifier"
"version": "v1"
```

Once the registry correctly prevented duplicate model-version records, repeated demo runs naturally collided with existing rows.

### Fix

Make the demo generate a unique model name per run:

```python
from uuid import uuid4

run_id = uuid4().hex[:8]
model_name = f"demo-classifier-{run_id}"
```

Then use `model_name` consistently across registration, activation, prediction, evaluation, comparison, traffic switching, and rollback.

### Engineering Takeaway

End-to-end demos should be repeatable. When database constraints are enforced correctly, demo scripts should either generate unique test resources or clean up their own state.

---

## 7. Host-generated artifacts were not visible inside Docker containers

### Symptom

The model registration response initially contained a Windows host path such as:

```text
C:\Users\...\AppData\Local\Temp\...\model_v1.joblib
```

The API service, however, runs inside a Linux Docker container and could not access this path.

### Root Cause

`demo.py` generated model artifacts on the Windows host, but the API and worker services consumed artifacts inside Docker containers.

A host-local path is not automatically visible inside a container.

### Fix

Create a shared artifact directory:

```text
./artifacts
```

Mount it into both `api` and `worker` services:

```yaml
volumes:
  - ./artifacts:/app/artifacts
```

Then separate host paths from container-visible paths in `demo.py`:

```python
host_model_v1_path = ARTIFACTS_DIR / "model_v1.joblib"
joblib.dump(clf_v1, host_model_v1_path)

container_model_v1_path = f"/app/artifacts/{host_model_v1_path.name}"
```

Use the container-visible path as the registered `artifact_uri`:

```python
json={
    "name": model_name,
    "version": "v1",
    "artifact_uri": container_model_v1_path,
}
```

### Engineering Takeaway

In Dockerized ML systems, artifact paths must be valid from the perspective of the service that loads them. A shared volume provides a simple local solution for making host-generated files available inside containers.

---

## 8. `WindowsPath` objects are not JSON serializable

### Symptom

During evaluation, the demo failed with:

```text
TypeError: Object of type WindowsPath is not JSON serializable
```

### Root Cause

`httpx` serializes `json=...` payloads using `json.dumps()`. Python `Path` objects such as `WindowsPath` are not JSON-native values.

In addition, even converting a Windows path to a string is not sufficient when the API runs inside Docker, because the container cannot access the Windows path directly.

### Fix

Create a container-visible dataset path:

```python
container_dataset_path = f"/app/artifacts/{dataset_path.name}"
```

Then pass that string in the JSON payload:

```python
json={
    "dataset_path": container_dataset_path,
}
```

### Engineering Takeaway

HTTP JSON payloads should only contain JSON-serializable values. In containerized systems, file paths should also be valid from the receiving service's runtime environment.

---

## 9. `/predict` failed because `pandas` was missing in the API container

### Symptom

The `/predict` endpoint returned `500 Internal Server Error`.

The demo client later failed when trying to access:

```python
data["predictions"]
```

### Root Cause

The API container was missing a runtime dependency required by the prediction adapter. The API logs showed:

```text
No module named 'pandas'
```

The local Python environment and Docker image had diverged.

### Fix

Add `pandas` to `requirements.txt`:

```text
pandas
```

Then rebuild the containers:

```bash
docker compose down
docker compose up --build -d
```

### Engineering Takeaway

A project can work in the local virtual environment but fail inside Docker if dependencies are not explicitly included in the container image. The Docker image should be treated as the source of truth for deployed runtime dependencies.

---

## 10. Success response assumptions caused secondary client-side errors

### Symptom

The demo client raised:

```text
KeyError: 'predictions'
```

when reading:

```python
data["predictions"]
```

### Root Cause

The client assumed that `/predict` always returned the success response schema:

```json
{
  "predictions": [...],
  "model_version": "...",
  "latency_ms": ...
}
```

When the server returned an error response, the expected `predictions` key did not exist.

### Fix

Check status codes before accessing success-only fields:

```python
resp = client.post("/predict", json=payload)

print("Status:", resp.status_code)
print("Raw text:", repr(resp.text))

if resp.status_code >= 400:
    raise RuntimeError(f"/predict failed: {resp.status_code} {resp.text}")

data = resp.json()
```

### Engineering Takeaway

Demo clients should not assume every response matches the success schema. Checking status codes makes failures easier to diagnose and prevents misleading secondary errors.

---

## 11. Broken artifact handling and rollback behavior

### Symptom

The demo intentionally registered and activated a broken model artifact:

```text
/nonexistent/bad_model.joblib
```

Prediction returned:

```text
422 Unprocessable Entity
```

with an error indicating the model artifact could not be found.

### Behavior Verified

The system correctly:

1. activated the broken version
2. failed prediction with a controlled 422 response
3. rolled back to a known-good version
4. restored successful prediction

### Engineering Takeaway

The demo validates more than the happy path. It also demonstrates a basic deployment safety workflow: controlled failure on invalid artifacts and rollback to a stable model version.

---

## Debugging Summary

While bringing up AtlasML locally, the main issues fell into five categories.

### 1. Migration and Docker setup

* Fixed Alembic import path resolution inside the API container.
* Replaced `localhost` database URLs with the Docker Compose Postgres service name.
* Synced SQLAlchemy constraints with Alembic migrations.

### 2. Registry correctness

* Repeated demo runs exposed duplicate `(model_name, version)` records.
* Added database-level uniqueness for model versions.
* Updated the demo to avoid fixed identifiers so it can be rerun safely.

### 3. Artifact accessibility

* Model artifacts generated on the Windows host were not visible inside Docker.
* Added a shared `artifacts/` volume and registered container-visible paths.
* Converted `Path` objects into JSON-serializable string URIs.

### 4. Inference runtime

* `/predict` initially failed because the Docker image missed `pandas`.
* Added the missing dependency and rebuilt the API container.
* Added status-code checks before parsing success response fields.

### 5. Reliability workflow

* Simulated a broken model artifact.
* Verified that prediction failed with a controlled 422 response.
* Rolled back to a stable version and restored successful inference.

---

## Final Takeaway

The main challenge was not a single bug, but making a generated ML infrastructure project work correctly across multiple runtime layers:

```text
demo.py client
  -> FastAPI API
  -> SQLAlchemy ORM
  -> Alembic migrations
  -> Postgres
  -> Docker networking
  -> shared artifact volume
  -> Redis/RQ worker path
  -> model adapter runtime
```

This debugging process clarified the separation between:

* persistent metadata in Postgres
* temporary model adapter caching in the API process
* transient task queues in Redis
* physical model artifacts mounted through Docker volumes
* client-side request/response handling

These fixes made the demo more reliable, repeatable, and closer to a real ML infrastructure workflow.