# AtlasML Agent Evals

## Goal

Agent evals measure task-level agent behavior rather than single-model output quality.

## Metrics

| Metric | Meaning |
|---|---|
| Task Success | Final answer contains the expected result |
| Tool Accuracy | Agent called the expected tool |
| Avg Steps | Average number of trace steps |
| Avg Latency | Average agent run duration |
| Avg Tokens | Average total LLM tokens |

## Dataset

The eval uses `data/agent_eval_tasks.jsonl`.

Each row includes:

- `task`
- `expected_tool`
- `expected_answer_contains`

## Interpretation

Agent evals help debug whether failures come from wrong tool selection, bad tool arguments, invalid final answers, or excessive latency/token usage.