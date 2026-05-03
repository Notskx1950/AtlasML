# AtlasML LLM Benchmarks

## Goal

This benchmark measures AtlasML's OpenAI-compatible LLM serving path.

It evaluates:

- short prompt latency
- long prompt latency
- structured JSON output reliability
- success rate
- p50 / p95 latency

## System Under Test

The benchmark calls AtlasML's `/predict` endpoint with an LLM-backed model registered through the model registry.

The LLM adapter uses an OpenAI-compatible chat completions API.

## Scenarios

| Scenario | Purpose |
|---|---|
| short_prompt | Baseline LLM request latency |
| long_prompt | Prompt length sensitivity |
| json_output | Structured output reliability |

## Results

| Scenario | Runs | Success Rate | Schema Valid Rate | Avg ms | P50 ms | P95 ms | Min ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_prompt | 10 | 1.00 | N/A | 855.11 | 615.85 | 994.11 | 515.73 | 2253.86 |
| long_prompt | 10 | 1.00 | N/A | 2663.82 | 2668.10 | 2995.14 | 2145.02 | 3145.17 |
| json_output | 10 | 1.00 | 1.00 | 2305.72 | 2147.41 | 2767.93 | 1696.77 | 3186.40 |

## Interpretation

The benchmark measures end-to-end LLM serving latency through AtlasML, including HTTP request handling, model registry resolution, LLM adapter execution, response parsing, and inference logging.

The JSON output scenario measures structured-output reliability by checking whether the response can be parsed as JSON and contains the required fields.

## Limitations

These measurements depend on the selected model, network conditions, provider latency, and local deployment environment. They should be interpreted as development-environment benchmark results rather than production capacity estimates.