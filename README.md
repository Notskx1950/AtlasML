# AtlasML

AtlasML is a lightweight ML/LLM serving and evaluation backend that demonstrates core AI infrastructure patterns: model versioning, validated artifact registration, synchronous and asynchronous inference, evaluation tracking, and observability.

## Quick Start

```bash
# Start all services
docker compose up -d

# Run migrations
docker compose exec api alembic upgrade head

# Run the demo
python demo.py
```

## Running Tests

```bash
pip install -r requirements.txt aiosqlite
pytest tests/ -v
```

## Architecture

- **FastAPI** API server with async PostgreSQL via SQLAlchemy 2.x
- **Redis + RQ** for async job processing (predictions and eval runs)
- **Model Registry** with version activation and rollback
- **Eval Framework** with classification and LLM metrics
- **Prometheus** metrics endpoint at `/metrics`
- **structlog** JSON-structured logging with request ID tracing

## API Overview

| Endpoint | Description |
|---|---|
| `POST /models/register` | Register a model version |
| `POST /models/{name}/activate` | Activate a version |
| `POST /models/{name}/deployments` records model activation and rollback as deployment events| 
| `POST /predict` | Synchronous inference |
| `POST /jobs/predict` | Async inference via RQ |
| `GET /jobs/{job_id}` | Job status |
| `POST /eval/run` | Start an eval run |
| `GET /eval/runs/{run_id}` | Get eval results |
| `GET /eval/compare` | Compare two versions |

## Diagram
``` text
                         ┌──────────────────────┐
                         │      demo.py         │
                         │  local test client   │
                         └──────────┬───────────┘
                                    │ HTTP
                                    │ localhost:8000
                                    ▼
┌──────────────────────────────────────────────────────────┐
│                    FastAPI API Server                    │
│                      api container                       │
│                                                          │
│  ┌────────────────┐   ┌────────────────┐                 │
│  │ registry API   │   │ inference API  │                 │
│  │ register/      │   │ predict        │                 │
│  │ activate       │   │                │                 │
│  └────────────────┘   └───────┬────────┘                 │
│                               │                          │
│                               ▼                          │
│                    ┌────────────────────┐                │
│                    │ RegistryStore      │                │
│                    │ adapter cache      │                │
│                    └─────────┬──────────┘                │
│                              │                           │
│                              ▼                           │
│                    ┌────────────────────┐                │
│                    │ ModelAdapter       │                │
│                    │ Sklearn / LLM      │                │
│                    └────────────────────┘                │
│                                                          │
│  ┌────────────────┐                                      │
│  │ eval API       │── enqueue job ───────┐               │
│  └────────────────┘                      │               │
└───────────────┬──────────────────────────┼───────────────┘
                │                          │
                │ SQLAlchemy               │ Redis/RQ
                ▼                          ▼
┌────────────────────────┐       ┌────────────────────────┐
│       Postgres         │       │         Redis          │
│                        │       │      job queue         │
│ - ModelVersion         │       └───────────┬────────────┘
│ - InferenceLog         │                   │
│ - EvalRun              │                   ▼
│ - EvalMetric           │       ┌────────────────────────┐
│ - JobRecord            │       │        Worker          │
└────────────────────────┘       │  run_eval / run_predict│
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │ /app/artifacts          │
                                │ shared Docker volume    │
                                │ model files + datasets  │
                                └─────────────────────────┘
                                ```