## Architecture

AtlasML is organized as a lightweight ML/LLM serving and evaluation platform. The API layer handles model registration, inference, and evaluation requests. PostgreSQL stores metadata and execution records, while Redis/RQ supports asynchronous background jobs.

```mermaid
flowchart LR
    Client[Client / User] --> API[FastAPI API Layer]

    API --> Registry[Registry API]
    API --> Inference[Inference API]
    API --> Eval[Evaluation API]

    Registry --> DB[(PostgreSQL Metadata DB)]
    Inference --> DB
    Eval --> DB

    Inference --> Queue[Redis / RQ Queue]
    Eval --> Queue

    Queue --> Worker[Background Worker]

    Inference --> Adapters[Model Adapters]
    Worker --> Adapters

    Adapters --> Artifacts[Model Artifacts<br/>joblib / LLM config / future model files]

    API --> Logs[Structured Logs]
    Worker --> Logs

    API --> Metrics[Prometheus Metrics]
    Worker --> Metrics
```

## Async Prediction Workflow

The async prediction path is used when inference should be submitted as a background job instead of being executed immediately in the API request.

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI API
    participant DB as PostgreSQL
    participant Q as Redis/RQ Queue
    participant W as Worker
    participant R as Registry
    participant M as Model Adapter

    C->>API: POST /jobs/predict
    API->>DB: Create JobRecord with queued status
    API->>Q: Enqueue prediction task
    API-->>C: Return job_id

    Q->>W: Dispatch prediction job
    W->>DB: Load job and model metadata
    W->>R: Resolve active model version
    R-->>W: Return active model artifact/config
    W->>M: Load model and run prediction
    M-->>W: Return prediction result

    W->>DB: Save result and update JobRecord
    C->>API: GET /jobs/{job_id}
    API->>DB: Read job status/result
    API-->>C: Return job status and result
```

