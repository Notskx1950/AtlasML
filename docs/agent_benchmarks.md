# AtlasML Agent Benchmarks

## Goal

This benchmark measures end-to-end agent execution through AtlasML.

It includes:

- LLM tool-call generation
- tool execution
- trace persistence
- final answer generation
- token and latency tracking

## Results

| Scenario | Runs | Success Rate | Avg ms | P50 ms | P95 ms | Avg Steps | Avg Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| calculator_agent_run | 5 | 1.00 | 4982.49 | 5453.67 | 5594.73 | 3 | 248 |

## Interpretation

Agent benchmarks measure workflow-level behavior rather than single model inference latency.

They help answer:

- Did the agent complete the task?
- Did it call the expected tool?
- How many steps did it take?
- How many tokens did it use?
- What was the end-to-end latency?