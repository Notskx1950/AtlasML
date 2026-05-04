# AtlasML Code Map

AtlasML is a FastAPI-based ML/LLM platform with model registry, inference, evaluation, background jobs, observability, and a minimal tool-calling agent runner.

This map reflects the codebase after the updates from `71d9ab620a521d015092dc6e5bb59f15920a964a` through `79636bb7db8275d1f61dfed65bf9320f21326b3a`.

## Project Structure

```text
AtlasML/
|-- alembic/                         # Database migration configuration
|   |-- env.py                       # Alembic environment and metadata wiring
|   |-- script.py.mako               # Alembic migration template
|   `-- versions/
|       |-- 001_initial_schema.py
|       |-- 25d8a45743ab_add_eval_run_reproducibility_fields.py
|       |-- d81e7fed3d81_add_deployment_events.py
|       |-- dc39ab989787_add_model_version_metadata.py
|       `-- 0c8a1adc90b4_add_agent_trace_tables.py
|
|-- app/
|   |-- api/                         # FastAPI route modules
|   |   |-- deps.py                  # Request-scoped DB session and optional API key dependency
|   |   |-- health.py                # /health and /ready
|   |   |-- registry.py              # /models registry, activation, deployments, stats
|   |   |-- inference.py             # /predict, /jobs/predict, /jobs/{job_id}
|   |   |-- eval.py                  # /eval/run, /eval/runs/{run_id}, /eval/compare
|   |   `-- agents.py                # /agents/run, trace lookup, run summary, stats
|   |
|   |-- agents/                      # Minimal tool-calling agent runtime
|   |   |-- runner.py                # AgentRunner orchestration and trace persistence
|   |   |-- schemas.py               # Tool-call request schema
|   |   `-- tools.py                 # calculator, echo_json, model_stats_lookup
|   |
|   |-- db/
|   |   |-- models.py                # SQLAlchemy ORM models
|   |   `-- session.py               # Lazy async and sync SQLAlchemy session factories
|   |
|   |-- eval/
|   |   |-- metrics.py               # Classification and LLM-oriented metric utilities
|   |   `-- runner.py                # Eval orchestration over datasets and adapters
|   |
|   |-- models/
|   |   |-- base.py                  # Shared ModelAdapter interface
|   |   |-- registry_store.py        # In-process adapter cache and adapter construction
|   |   |-- sklearn_adapter.py       # sklearn/joblib model serving
|   |   |-- llm_adapter.py           # OpenAI-compatible chat completions adapter
|   |   `-- agent_tool_adapter.py    # Adapter support for tool-oriented model workflows
|   |
|   |-- utils/
|   |   `-- hashing.py               # Dataset and artifact hashing helpers
|   |
|   |-- workers/
|   |   `-- tasks.py                 # RQ tasks for async prediction and eval
|   |
|   |-- config.py                    # Environment-backed settings
|   `-- main.py                      # FastAPI app factory, middleware, metrics, router registration
|
|-- data/
|   `-- agent_eval_tasks.jsonl       # Task-level agent eval dataset
|
|-- docs/
|   |-- architecture_diagram.md      # Architecture diagrams and workflows
|   |-- code_map.md                  # This source map
|   |-- api_errors.md                # API error contract notes
|   |-- benchmarks.md                # API latency benchmark results
|   |-- llm_benchmarks.md            # LLM serving benchmark results
|   |-- edge_benchmark.md            # Edge optimization benchmark results
|   |-- agent_benchmarks.md          # Agent benchmark results
|   |-- agent_evals.md               # Agent eval analysis
|   `-- debugging_notes.md           # Development notes
|
|-- examples/
|   `-- langchain_agent_demo.py      # Example agent integration
|
|-- scripts/
|   |-- benchmark_api.py             # Service latency benchmark
|   |-- benchmark_llm.py             # LLM serving benchmark
|   |-- benchmark_edge.py            # PyTorch, quantization, TorchScript, ONNX benchmark
|   |-- benchmark_agent.py           # Agent workflow benchmark
|   `-- run_agent_eval.py            # Task-level agent eval runner
|
|-- tests/
|   |-- conftest.py                  # Test fixtures
|   |-- test_auth.py                 # API key behavior
|   |-- test_health.py               # Health and readiness behavior
|   |-- test_registry.py             # Registry behavior and error contracts
|   |-- test_inference.py            # Prediction paths
|   |-- test_stat_api.py             # Model stats API
|   |-- test_eval.py                 # Eval API and runner behavior
|   |-- test_eval_reproducibility.py # Eval reproducibility metadata
|   |-- test_agent_traces.py         # Agent trace persistence and lookup
|   |-- test_agent_stats.py          # Agent stats API
|   |-- test_agents_runner.py        # AgentRunner and built-in tools
|   |-- test_agent_eval.py           # Agent eval scoring behavior
|   |-- test_deployment.py           # Deployment event behavior
|   `-- test_hashing.py              # Hashing utilities
|
|-- artifacts/                       # Local model artifacts and generated benchmark outputs
|-- .env.example                     # Example local configuration
|-- docker-compose.yml               # Local Postgres, Redis, API, and worker stack
|-- Dockerfile                       # API image definition
|-- requirements.txt                 # Core runtime and test dependencies
|-- requirements-edge.txt            # Optional edge benchmark dependencies
|-- demo.py                          # End-to-end demo client
`-- README.md
```

## Runtime Entry Points

| File | Responsibility |
|---|---|
| `app/main.py` | Builds the FastAPI application, configures structlog, adds request ID middleware, exposes Prometheus metrics, and registers all routers. |
| `app/config.py` | Loads database, Redis, logging, API key, timeout, and OpenAI-compatible LLM settings from environment variables. |
| `app/api/deps.py` | Provides `get_db()` for async request sessions and `require_api_key()` for protected routes. |
| `app/db/session.py` | Creates lazy async and sync SQLAlchemy session factories. FastAPI uses async sessions; RQ workers use sync sessions. |
| `app/workers/tasks.py` | Executes queued prediction and eval jobs, updates DB-backed status, and logs outcomes. |

## API Routes

| Module | Routes | Notes |
|---|---|---|
| `app/api/health.py` | `GET /health`, `GET /ready` | `/ready` checks Postgres and Redis and returns 503 when a dependency is unavailable. |
| `app/api/registry.py` | `POST /models/register`, `GET /models/{name}`, `GET /models/{name}/active`, `POST /models/{name}/activate`, `GET /models/{name}/deployments`, `GET /models/{name}/stats` | Registration validates local non-LLM artifacts and stores SHA256 hashes. Activation records deployment events. Stats summarize inference logs, latency, errors, and token usage. |
| `app/api/inference.py` | `POST /predict`, `POST /jobs/predict`, `GET /jobs/{job_id}` | Sync predictions log latency, errors, schema validity, and optional LLM token usage. Async predictions create `JobRecord` rows and enqueue RQ jobs. |
| `app/api/eval.py` | `POST /eval/run`, `GET /eval/runs/{run_id}`, `GET /eval/compare` | Eval launch records reproducibility metadata and enqueues RQ work. Comparison reads latest completed metrics for two versions. |
| `app/api/agents.py` | `POST /agents/run`, `GET /agents/runs/{run_id}`, `GET /agents/runs/{run_id}/trace`, `GET /agents/stats` | Runs the minimal agent loop, returns trace detail, and aggregates agent observability. |

Protected routes use `require_api_key` when `API_KEY` is configured. The current protected surfaces are model registration, model activation, sync prediction, async prediction enqueue, and eval launch.

## Database Models

| Model | Purpose |
|---|---|
| `ModelVersion` | Registered model metadata, artifact URI, runtime config, activity flag, framework, task type, status, tags, and artifact hash. |
| `DeploymentEvent` | Activation and rollback history for model versions. |
| `InferenceLog` | Per-request inference latency, status, error type, token counts, schema validity, and tool success flag. |
| `EvalRun` | Eval run state, dataset identity, dataset hash, row count, git commit, config snapshot, duration, and status. |
| `EvalMetric` | Metric name/value rows attached to an eval run. |
| `JobRecord` | Async prediction job lifecycle, result, and error state. |
| `AgentRun` | Full agent task execution, model reference, status, output, error, token counts, and duration. |
| `AgentStep` | Ordered agent trace steps, including LLM calls, tool calls, final answers, and errors. |
| `ToolInvocation` | Structured tool call input, output, status, error, latency, and step linkage. |

## Model and Adapter Layer

The adapter layer is centered on `ModelAdapter` in `app/models/base.py`.

- `RegistryStore` caches adapters by `(name, version)` and builds adapters from `ModelVersion.runtime_config`.
- `SklearnAdapter` loads local `.joblib` artifacts.
- `LLMAdapter` calls an OpenAI-compatible chat completions API and exposes `last_input_tokens` and `last_output_tokens` for metrics.
- Non-LLM model registration validates local artifact existence and stores a SHA256 hash.

## Evaluation Flow

`app/eval/runner.py` loads the target model adapter, reads the dataset, runs predictions, computes metrics from `app/eval/metrics.py`, and writes `EvalMetric` rows. The eval API records `git_commit` and `config_snapshot` so results can be tied back to code and runtime configuration.

## Agent Flow

`AgentRunner` in `app/agents/runner.py` follows a three-step happy path:

1. Ask an LLM-backed adapter to return JSON with `tool_name` and `arguments`.
2. Execute one built-in tool from `app/agents/tools.py`.
3. Ask the model to generate a final answer from the tool result.

Each stage writes trace records to `AgentRun`, `AgentStep`, and `ToolInvocation`. Failures are captured as failed runs with error metadata and an error step.

## Observability and Benchmarks

- Service health: `GET /health`
- Dependency readiness: `GET /ready`
- Prometheus metrics: `GET /metrics`
- Model observability: `GET /models/{name}/stats`
- Agent observability: `GET /agents/stats` and `GET /agents/runs/{run_id}/trace`
- API benchmark: `scripts/benchmark_api.py`
- LLM benchmark: `scripts/benchmark_llm.py`
- Edge benchmark: `scripts/benchmark_edge.py`
- Agent benchmark: `scripts/benchmark_agent.py`
- Agent eval runner: `scripts/run_agent_eval.py`
