# AtlasML Agent Evals

## Goal

Agent evals measure task-level agent behavior rather than single-model prediction quality.

Unlike `/predict` benchmarks, agent evals evaluate the full workflow:

- whether the agent chooses the correct tool
- whether the tool call is structured correctly
- whether the final answer contains the expected result
- how many steps the agent takes
- how much latency and token usage each task requires
- which task categories expose failure modes

## Dataset

The eval uses:

```text
data/agent_eval_tasks.jsonl
````

The task set includes multiple categories:

| Category             | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `calculator_basic`   | Tests arithmetic tool use                                          |
| `echo_json`          | Tests structured JSON argument passing                             |
| `model_stats_lookup` | Tests whether the agent can call an AtlasML observability tool     |
| `tool_selection`     | Tests whether the agent chooses the correct tool from task wording |
| `failure_cases`      | Tests robustness and controlled failure behavior                   |

Each row contains fields such as:

```json
{
  "category": "calculator_basic",
  "difficulty": "easy",
  "task": "Calculate 18 * 23.",
  "expected_tool": "calculator",
  "expected_answer_contains": "414"
}
```

## Metrics

| Metric         | Meaning                                   |
| -------------- | ----------------------------------------- |
| Task Success   | Final answer contains the expected result |
| Tool Accuracy  | Agent called the expected tool            |
| Avg Steps      | Average number of trace steps per run     |
| Avg Latency ms | Average end-to-end agent run duration     |
| Avg Tokens     | Average total LLM token usage             |

## Overall Results

| Task Set         | Runs | Task Success | Tool Accuracy | Avg Steps | Avg Latency ms | Avg Tokens |
| ---------------- | ---: | -----------: | ------------: | --------: | -------------: | ---------: |
| agent_eval_tasks |   15 |         0.47 |          0.40 |      1.67 |        1553.27 |     103.00 |

## Category Breakdown

| Category           | Runs | Task Success | Tool Accuracy | Avg Steps | Avg Latency ms | Avg Tokens |
| ------------------ | ---: | -----------: | ------------: | --------: | -------------: | ---------: |
| calculator_basic   |    5 |         0.20 |          0.20 |      1.40 |        1103.00 |      84.00 |
| echo_json          |    3 |         0.67 |          0.67 |      2.33 |        1598.00 |     131.67 |
| failure_cases      |    2 |         1.00 |          0.50 |      1.00 |         869.00 |      76.50 |
| model_stats_lookup |    2 |         0.00 |          0.00 |      1.00 |        2484.50 |      87.50 |
| tool_selection     |    3 |         0.67 |          0.67 |      2.33 |        2094.33 |     134.00 |

## Interpretation

The current agent eval results show that the minimal AgentRunner can execute some tool-based workflows, but tool selection and structured tool-call reliability are not yet stable across task categories.

The overall task success rate is `0.47`, and tool accuracy is `0.40`. This suggests that the primary bottleneck is not the Python tool execution layer itself, but the LLM's ability to consistently produce the correct tool name and arguments in the expected JSON format.

The average step count is `1.67`. A fully successful single-tool agent run usually follows a three-step pattern:

```text
1. LLM generates a tool call
2. AtlasML executes the selected tool
3. LLM generates the final answer from the tool result
```

Since the average step count is much lower than `3`, many failures likely happen early during tool-call generation, parsing, or validation before the workflow reaches tool execution and final response generation.

## Category Analysis

### `calculator_basic`

| Runs | Task Success | Tool Accuracy | Avg Steps | Avg Tokens |
| ---: | -----------: | ------------: | --------: | ---------: |
|    5 |         0.20 |          0.20 |      1.40 |      84.00 |

The calculator category currently performs poorly, despite the standalone `calculator_agent_run` benchmark succeeding.

This suggests that the agent can solve a specific calculator prompt, but is not yet robust across varied arithmetic task wording.

Likely failure modes include:

* LLM returns natural language instead of pure tool-call JSON
* wrong or missing `calculator` tool name
* invalid `arguments` structure
* expression contains extra text, such as `"18 * 23 = 414"` instead of `"18*23"`
* arithmetic expression includes unsupported syntax

This category should improve significantly with stricter tool-call prompting and examples.

---

### `echo_json`

| Runs | Task Success | Tool Accuracy | Avg Steps | Avg Tokens |
| ---: | -----------: | ------------: | --------: | ---------: |
|    3 |         0.67 |          0.67 |      2.33 |     131.67 |

The JSON echo category is more stable than calculator tasks, but still not perfect.

The average step count of `2.33` indicates that some runs complete most of the expected workflow, but not all tasks reach the full three-step pattern.

Likely failure modes include:

* wrong tool selection
* incorrect nesting of the JSON payload
* final answer does not include the expected value
* LLM modifies the JSON instead of echoing it unchanged

This category is useful for testing structured argument passing.

---

### `model_stats_lookup`

| Runs | Task Success | Tool Accuracy | Avg Steps | Avg Tokens |
| ---: | -----------: | ------------: | --------: | ---------: |
|    2 |         0.00 |          0.00 |      1.00 |      87.50 |

The model stats lookup category currently fails completely.

The average step count is `1.00`, which strongly suggests that failures happen at the initial tool-call generation or parsing stage.

Likely failure modes include:

* wrong tool name, such as `get_model_stats` or `lookup_model_stats`
* wrong argument name, such as `model` instead of `model_name`
* natural-language answer instead of JSON tool call
* invalid tool-call JSON

This is the clearest signal that the current tool schema prompt is not strict enough for platform-observability tools.

---

### `tool_selection`

| Runs | Task Success | Tool Accuracy | Avg Steps | Avg Tokens |
| ---: | -----------: | ------------: | --------: | ---------: |
|    3 |         0.67 |          0.67 |      2.33 |     134.00 |

The tool selection category shows partial success.

This means the agent has some ability to infer the correct tool from task wording, but it is not reliable enough yet.

This category is important because real agent workloads often require selecting between multiple tools based on ambiguous user intent.

---

### `failure_cases`

| Runs | Task Success | Tool Accuracy | Avg Steps | Avg Tokens |
| ---: | -----------: | ------------: | --------: | ---------: |
|    2 |         1.00 |          0.50 |      1.00 |      76.50 |

The failure case category should be interpreted carefully.

Because some failure-case rows may have `expected_answer_contains = null`, the scoring helper may count them as successful by default. Therefore, the `1.00` task success rate does not necessarily mean the agent handled failures correctly.

Failure cases should eventually be scored separately with a metric such as:

```text
controlled_failure_rate
```

rather than being mixed into normal task success.

## Key Takeaways

* The AgentRunner can execute successful tool-based workflows, but reliability varies by task category.
* Low average step count indicates many failures occur before full tool execution and final answer generation.
* Tool selection and JSON tool-call formatting are the main current bottlenecks.
* `model_stats_lookup` is the weakest category and should be prioritized for prompt/schema improvements.
* `failure_cases` should be separated from normal task-success scoring in future evals.
* The eval framework is already useful because it exposes concrete failure modes instead of only reporting happy-path success.

## Recommended Next Improvements

### 1. Strengthen the tool-call prompt

The tool-call prompt should force a strict JSON format:

```json
{
  "tool_name": "calculator",
  "arguments": {
    "expression": "18*23"
  }
}
```

It should explicitly say:

```text
Return only valid JSON.
Do not include markdown.
Do not include explanation.
Choose tool_name from: calculator, echo_json, model_stats_lookup.
```

### 2. Add tool examples

Examples should be included in the tool-call prompt:

```text
Task: Calculate 18 * 23.
Output: {"tool_name":"calculator","arguments":{"expression":"18*23"}}

Task: Echo this JSON: {"status":"ok"}.
Output: {"tool_name":"echo_json","arguments":{"payload":{"status":"ok"}}}

Task: Look up stats for model llm-benchmark.
Output: {"tool_name":"model_stats_lookup","arguments":{"model_name":"llm-benchmark"}}
```

### 3. Add tool and argument alias normalization

The runner can normalize common LLM variants:

```python
TOOL_ALIASES = {
    "get_model_stats": "model_stats_lookup",
    "lookup_model_stats": "model_stats_lookup",
    "model_stats": "model_stats_lookup",
}
```

Argument aliases:

```python
ARG_ALIASES = {
    "model": "model_name",
    "name": "model_name",
}
```

### 4. Separate failure-case scoring

Future eval output should separate:

```text
normal_task_success
tool_accuracy
controlled_failure_rate
```

This avoids failure cases inflating or distorting normal task-success metrics.

### 5. Inspect failed traces

For failed runs, inspect:

```text
GET /agents/runs/{run_id}/trace
```

The most useful fields are:

* raw LLM tool-call response
* parsed tool name
* parsed arguments
* tool invocation status
* error step output
* final answer, if any

## Summary

This eval shows that AtlasML now has a working task-level agent evaluation loop.

The current agent is not yet highly reliable, but the infrastructure is doing its job: it reveals where agent workflows succeed, where they fail, and which categories need improvement.

The next iteration should focus on stricter structured tool-call prompting, alias normalization, and separate robustness scoring.
