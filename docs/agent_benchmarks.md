# AtlasML Agent Benchmarks

## Goal

This benchmark measures end-to-end agent execution through AtlasML.

Unlike single-model inference benchmarks, this benchmark measures a full agent workflow:

- LLM tool-call generation
- structured tool selection
- tool execution
- trace persistence
- final answer generation
- token and latency tracking

The goal is to evaluate whether AtlasML can run and observe agent workflows, not just call an LLM once.

## System Under Test

The benchmark calls:

```text
POST /agents/run
GET /agents/runs/{run_id}/trace
````

Each successful agent run follows a three-step pattern:

```text
1. LLM generates a structured tool call
2. AtlasML executes the selected tool
3. LLM generates a final answer from the tool result
```

## Scenarios

| Scenario                       | Purpose                                                        |
| ------------------------------ | -------------------------------------------------------------- |
| `calculator_agent_run`         | Tests arithmetic tool use and final answer generation          |
| `echo_json_agent_run`          | Tests structured JSON tool argument passing                    |
| `model_stats_lookup_agent_run` | Tests whether the agent can call an AtlasML observability tool |

## Results

| Scenario                     | Runs | Success Rate |  Avg ms |  P50 ms |  P95 ms | Avg Steps | Avg Tokens |
| ---------------------------- | ---: | -----------: | ------: | ------: | ------: | --------: | ---------: |
| calculator_agent_run         |    5 |         1.00 | 3711.45 | 3722.39 | 3731.56 |         3 |        248 |
| echo_json_agent_run          |    5 |         1.00 | 1375.34 | 1305.98 | 1502.77 |         3 |      128.8 |
| model_stats_lookup_agent_run |    5 |         0.00 | 1569.54 | 1512.07 | 1680.04 |         1 |         88 |

## Interpretation

The benchmark shows that AtlasML can successfully execute complete three-step agent workflows for calculator and JSON echo scenarios. These successful runs include LLM tool-call generation, tool execution, final answer generation, token accounting, and trace persistence.

The `calculator_agent_run` scenario achieved a 100% success rate with an average of 3 steps, which matches the expected agent workflow: LLM tool call, tool execution, and final response. It used more tokens than the JSON scenario, likely because the final answer generation step requires more explanation.

The `echo_json_agent_run` scenario also achieved a 100% success rate with 3 average steps, showing that the agent can pass structured JSON arguments through a tool and produce a final response.

The `model_stats_lookup_agent_run` scenario failed in this run, with an average of only 1 step. This suggests that failures happen early, likely during structured tool-call generation or parsing, before tool execution and final answer generation. This is useful because it shows that the trace system can reveal where the agent workflow breaks.

## Key Takeaways

* AtlasML can execute and trace successful multi-step agent workflows.
* Successful calculator and JSON tasks follow the expected 3-step agent pattern.
* Token usage differs significantly by task type.
* The model stats lookup task currently exposes a tool-selection or tool-call-formatting weakness.
* Agent benchmark results are useful not only for success measurement, but also for failure diagnosis.

## Failure Analysis

The failed `model_stats_lookup_agent_run` scenario suggests that the current tool-calling prompt may not constrain the LLM strongly enough.

Likely failure modes include:

* wrong tool name, such as `get_model_stats` instead of `model_stats_lookup`
* wrong argument name, such as `model` instead of `model_name`
* invalid JSON output
* natural-language explanation instead of pure tool-call JSON

The next improvement should strengthen the tool-call schema prompt and optionally add tool/argument alias normalization.

## Limitations

These results are local development measurements and depend on:

* selected LLM provider
* network conditions
* prompt wording
* tool schema clarity
* local API and database overhead

This benchmark is not a production capacity estimate. It is intended to measure agent workflow behavior, traceability, and failure modes in AtlasML.

## Next Improvements

* Add stricter tool-call prompting with explicit JSON examples.
* Add tool alias normalization for common tool-name variants.
* Add argument alias normalization, for example `model` → `model_name`.
* Separate normal task success from controlled failure-case scoring.
* Add category-level benchmark results for calculator, JSON, model stats, and tool-selection tasks.
