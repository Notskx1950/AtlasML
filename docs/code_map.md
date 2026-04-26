# AtlasML

> A lightweight ML model registry, inference, and evaluation service.

## Overview

AtlasML is a modular machine learning service prototype that supports model registration, inference, evaluation, and background task execution. The project is organized around a FastAPI application with database-backed model metadata, adapter-based model loading, and an evaluation workflow.

## Project Structure

```text
AtlasML/
├── alembic/                  # Database migration configuration
│   ├── env.py                # Environmental setting of alembic, decide online or offline migrations, main entrance of calling Alembic
│   ├── script.py.mako        # Template used by Alembic to generate new migration files
│   └── versions/
│       └── 001_initial_schema.py # Initial migration that creates the database tables
│
├── app/                      # Core application package
│   ├── api/                  # API route definitions
│   │   ├── deps.py           # API routes use FastAPI dependencies to receive request-scoped async SQLAlchemy sessions.
│   │   ├── eval.py           # API routes for evaluation and comparison
│   │   ├── inference.py      # API routes for model inference, mainly for prediction and async prediction
│   │   └── registry.py       # API routes for model registration and activation
│   │
│   ├── db/                   # Database models and session management
│   │   ├── models.py         # SQLAlchemy ORM models for model versions, inference logs, eval runs, metrics, and jobs
│   │   └── session.py        # Defines lazy async and sync SQLAlchemy engine/session factories.
│   │
│   ├── eval/                 # Evaluation logic and metrics
│   │   ├── metrics.py        # Functions to compute evaluation metrics, like accuracy, f1 macro, f1 weighted, or LLM metrics.
│   │   └── runner.py         # Orchestrates evaluation runs. It loads the target model version and evaluation dataset, runs predictions through the model adapter, calls metric utilities from `metrics.py`, and returns structured evaluation results.
│   │
│   ├── models/               # Model adapter abstractions and registry logic
│   │   ├── base.py           # Defines the shared ModelAdapter interface used by all model adapters
│   │   ├── llm_adapter.py    # Adapter for OpenAI-compatible LLM-style models with structured output handling
│   │   ├── registry_store.py # In-memory registry/cache for loading and resolving registered model adapters
│   │   └── sklearn_adapter.py # Adapter for loading and serving sklearn/joblib model artifacts
│   │
│   ├── workers/              # Background job definitions
│   │   └── tasks.py          # Executes queued predict/eval tasks and updates DB-backed job status
│   │
│   ├── config.py             # Application configuration
│   ├── main.py               # FastAPI application entrypoint
│   └── __init__.py
│
├── artifacts/                # Example artifacts and test data
│   ├── model_v1.joblib
│   ├── model_v2.joblib
│   └── test_dataset.jsonl
│
├── docs/                     # Internal notes and documentation
│   ├── debugging_notes.md
│   └── code_map.md
│
├── tests/                    # Test suite
│   ├── conftest.py
│   ├── test_eval.py
│   ├── test_inference.py
│   ├── test_registry.py
│   └── __init__.py
│
├── .env.example              # Example environment variables
├── .gitignore
├── alembic.ini
├── demo.py                   # A complete demo to demonstrate current api features
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Components

### API Layer

The API layer is implemented with FastAPI and exposes the main user-facing workflows of the system.

It currently provides endpoints for:

- registering model versions
- activating a model version
- running synchronous inference
- submitting asynchronous prediction jobs
- checking job status
- starting evaluation runs
- retrieving evaluation results
- comparing evaluation metrics across model versions

The API routes live under `app/api/`. Database access is provided through shared FastAPI dependencies in `app/api/deps.py`, where each request receives a request-scoped async SQLAlchemy session.

At a high level, the API layer is responsible for receiving requests, validating payloads, creating or reading database records, and delegating longer-running work to model adapters, evaluation runners, or background workers.

---

### Database Layer

The database layer uses SQLAlchemy ORM models to persist system metadata and execution state.

The database does not store model artifacts directly. Instead, it stores metadata such as model names, versions, artifact paths, inference logs, evaluation runs, evaluation metrics, and async job records.

Core database concepts include:

- `ModelVersion`: stores registered model metadata, including model name, version, artifact URI, and runtime configuration
- `InferenceLog`: records inference requests, outputs, and failure information
- `EvalRun`: tracks an evaluation run for a specific model version and dataset
- `EvalMetric`: stores metrics produced by evaluation runs
- `JobRecord`: tracks asynchronous prediction job status, results, and errors

Database session management is centralized in `app/db/session.py`, which provides both async and sync SQLAlchemy session factories. FastAPI routes use async sessions, while RQ worker tasks use sync sessions.

Alembic is used for database schema migration.

---

### Model Registry and Adapter Layer

The model registry and adapter layer provide a unified way to register, load, and execute different types of models.

A model version is first registered through the API and stored in the database as metadata. The actual model artifact, such as a `.joblib` file, is stored separately under the artifact path.

The adapter layer is built around a shared `ModelAdapter` interface defined in `app/models/base.py`. Concrete adapters implement this interface for specific model types.

Current adapters include:

- `SklearnAdapter`: loads and serves sklearn/joblib model artifacts
- `LLMAdapter`: provides an OpenAI-compatible LLM-style adapter with structured output handling

The registry store helps resolve registered model versions and build the correct adapter based on the model artifact and runtime configuration.

This design separates model metadata management from model execution logic, making it easier to support heterogeneous model types in the future.

---

### Evaluation Workflow

The evaluation workflow is responsible for running models against evaluation datasets and computing metrics.

The main orchestration logic lives in `app/eval/runner.py`. The eval runner coordinates the evaluation process by:

1. loading the selected model adapter
2. loading or reading the evaluation dataset
3. running predictions
4. computing metrics through utilities in `app/eval/metrics.py`
5. writing evaluation results and metrics back to the database

The metrics module contains reusable metric functions such as classification metrics and LLM-oriented evaluation helpers.

This design separates "how to run an evaluation" from "how to compute a metric." The runner orchestrates the workflow, while `metrics.py` provides the metric calculation utilities.

---

### Workers

AtlasML uses Redis/RQ workers to execute long-running tasks outside the HTTP request-response path.

The worker task definitions live in `app/workers/tasks.py`.

Current worker tasks include:

- `run_predict(...)`: executes queued prediction jobs
- `run_eval(...)`: executes queued evaluation jobs

For asynchronous prediction, the API creates a `JobRecord`, enqueues a task into Redis/RQ, and immediately returns a `job_id`. A separate worker process then picks up the job, loads the requested model version, builds the adapter, runs prediction, and updates the job status and result in the database.

For evaluation jobs, the worker loads the requested model version, builds the adapter, and delegates the evaluation workflow to `EvalRunner`.

This architecture allows the API service to stay responsive while heavier prediction and evaluation workloads run in a separate worker container.

---

### Observability and Logging

The project includes basic structured logging through `structlog`. Worker tasks log successful and failed prediction/evaluation jobs with structured fields such as `job_id`, `run_id`, model name, metrics, and error messages.

The current implementation provides a foundation for observability, but there is room to extend it with request tracing, latency tracking, per-model metrics, and queue-level monitoring.

---

### Local Development and Deployment

The project includes Docker and Docker Compose configuration for local development.

The expected local stack includes:

- FastAPI application container
- PostgreSQL database
- Redis queue
- RQ worker container

The API container handles HTTP requests, while the worker container listens to Redis/RQ and executes background prediction or evaluation jobs.

A typical async workflow is:

```text
Client
  -> FastAPI API
  -> Redis/RQ queue
  -> Worker container
  -> Model adapter / Eval runner
  -> PostgreSQL status/result update
