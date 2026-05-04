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

## Token Metrics

AtlasML exposes token usage for LLM-backed predictions:

- `input_tokens`
- `output_tokens`
- `total_tokens`

The LLM benchmark reports average token usage per scenario, which helps explain latency differences between short prompts, long prompts, and structured JSON output.

## Scenarios

| Scenario | Purpose |
|---|---|
| short_prompt | Baseline LLM request latency |
| long_prompt | Prompt length sensitivity |
| json_output | Structured output reliability |

## Results

| Scenario | Runs | Success Rate | Schema Valid Rate | Avg ms | P50 ms | P95 ms | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_prompt | 10 | 1.00 | N/A | 636.96 | 595.52 | 823.36 | 14.00 | 24.00 | 38.00 |
| long_prompt | 10 | 1.00 | N/A | 2863.60 | 2778.07 | 3192.62 | 44.00 | 128.00 | 172.00 |
| json_output | 10 | 1.00 | 1.00 | 3348.25 | 2749.78 | 5782.39 | 33.00 | 88.00 | 121.00 |


## Interpretation

The benchmark measures end-to-end LLM serving latency through AtlasML, including HTTP request handling, model registry resolution, LLM adapter execution, response parsing, and inference logging.

The JSON output scenario measures structured-output reliability by checking whether the response can be parsed as JSON and contains the required fields.

## Limitations

These measurements depend on the selected model, network conditions, provider latency, and local deployment environment. They should be interpreted as development-environment benchmark results rather than production capacity estimates.