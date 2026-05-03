# AtlasML Service Benchmarks

## Goal

This benchmark measures common AI infrastructure backend paths in AtlasML:

- liveness and readiness checks
- synchronous prediction latency
- async prediction enqueue overhead
- eval enqueue overhead
- model-level observability query latency

## Environment

| Item | Value |
|---|---|
| Machine | Local development machine |
| Python | 3.12 |
| API | FastAPI / Uvicorn |
| Database | Postgres via Docker Compose |
| Queue | Redis / RQ |
| Runs per scenario | 50 |

## Results

| Scenario | Runs | Avg ms | P50 ms | P95 ms | Min ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|
| health check | 50 | 1.37 | 1.20 | 1.48 | 1.12 | 8.32 |
| readiness check | 50 | 3.80 | 3.70 | 3.98 | 3.53 | 6.42 |
| sync predict | 50 | 48.27 | 47.94 | 49.68 | 46.96 | 54.28 |
| async predict enqueue | 50 | 48.67 | 48.09 | 51.75 | 47.55 | 52.94 |
| eval enqueue | 50 | 49.44 | 48.31 | 52.05 | 47.62 | 52.48 |
| model stats query | 50 | 2.51 | 2.31 | 2.63 | 2.17 | 10.40 |

## Interpretation

- `/health` measures API process liveness.
- `/ready` includes dependency checks for Postgres and Redis.
- `/predict` measures end-to-end synchronous inference latency.
- `/jobs/predict` measures queue enqueue overhead.
- `/eval/run` measures eval orchestration launch overhead.
- `/models/{name}/stats` measures observability query latency over inference logs.

## Limitations

These numbers are local development measurements and are not production capacity estimates.
They are intended to compare relative behavior across API paths.