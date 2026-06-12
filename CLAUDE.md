# llm-inference-bench — Claude Code context

## Overview

Two-layer LLM inference benchmarking tool:

1. **Load benchmarking** (`collect/run_bench.py`) — TTFT/throughput/latency under concurrent load
2. **Quality-aware evaluation** (`evaluate/run_eval.py` + `analyze/deployment_advisor.py`) — production deployment recommendation balancing latency, cost, and quality

The existing pipeline (`report.py`, `playbook/advisor.py`) is untouched by the quality feature — it is purely additive.

## Running tests

A globally installed pytest plugin tries to bind a socket in this environment. Bypass it:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

44 tests across two files:

- `tests/test_run_eval.py` — 17 tests: dataset loading, score normalization, metric selection, sidecar writing
- `tests/test_deployment_advisor.py` — 27 tests: load_deployment, compute_tradeoff, recommend, render

## Key files

| File | Role |
| --- | --- |
| `collect/run_bench.py` | Async load benchmark — OpenAI-compatible `/v1/completions` |
| `evaluate/run_eval.py` | Offline quality evaluator — DeepEval + LLM-judge |
| `analyze/deployment_advisor.py` | Deployment decision engine — 4 pure functions + CLI |
| `analyze/report.py` | Markdown report from latency results (stdlib only) |
| `playbook/advisor.py` | vLLM config recommendation (stdlib only) |
| `data/generate_synthetic.py` | Synthetic reference data |
| `datasets/*.jsonl` | Eval datasets — `schema_version: 1`, workloads: chat/rag/long_context |
| `results/quality/` | Quality sidecars written by `run_eval.py` |
| `results/synthetic/` | Committed reference data |
| `results/real/` | Gitignored — populated by `run_bench.py` |

## Architecture invariants

### Quality sidecar coupling

Each quality sidecar (`results/quality/<tag>.json`) carries a `latency_tag` backlink to the latency result it was paired with. Hard errors:

- `latency_tag` in sidecar ≠ the tag being loaded — stale sidecar can silently corrupt a recommendation
- `meta.model` in sidecar ≠ `meta.model` in latency result — different model, incomparable

### Cross-profile dataset validation

`compute_tradeoff` hard-errors if profiles carry different `_dataset` values. Quality scores from different eval sets are not comparable. This check lives in `compute_tradeoff`, not in `load_deployment`.

### Dataset schema

`load_dataset` in `run_eval.py` validates:

- Required fields: `schema_version`, `id`, `workload`, `prompt`, `expected`
- `schema_version` must equal `1` — any other value is a hard error
- Valid workloads: `"chat"`, `"rag"`, `"long_context"`
- RAGAS fields (`contexts`, `ground_truth`) are V2 only — no V1 code or dependency

### DeploymentProfile contract

`load_deployment` flattens the nested latency JSON into a normalized in-memory dict:

- `metrics.ttft_ms.p50` → `latency.ttft_ms_p50`
- `metrics.ttft_ms.p95` → `latency.ttft_ms_p95`
- `metrics.throughput_tokens_per_sec` → `latency.throughput_tokens_per_sec`
- `quality` is `None` when no sidecar exists (warn, not error)
- `_dataset` is `None` when no sidecar; set to `meta.dataset` from the sidecar when one is loaded

### Real overrides synthetic

In `load_deployment`, `latency_dirs` is searched in order and **later directories win**. The default order is `[results/synthetic, results/real]`, so real measurements silently override synthetic reference data for the same tag. This matches `report.py`'s behavior.

## Known gotchas

**DeepEval env var mutation.** When `--eval-endpoint` or `--eval-token` are provided, `run_deepeval()` sets `OPENAI_API_KEY` and/or `OPENAI_BASE_URL` as process-level environment variables before running metrics. Safe for the CLI (runs to completion and exits), but would be a footgun if the function were called from a long-running server or a test suite that parallelizes eval runs.

**Hallucination normalization differs by evaluator path.** The LLM-judge prompt scores hallucination 1–5 where `5=none` (already higher-is-better). After dividing by 5, the score is in [0,1] with higher=better — `normalize_score("hallucination", ...)` is NOT applied. On the DeepEval path, `HallucinationMetric.score` returns a rate (lower is better), so `normalize_score` inverts it via `1 - score`. The two paths are intentionally asymmetric.

**DeepEval is a lazy import.** `run_deepeval` imports DeepEval inside the function body, not at module level. This keeps the 44 unit tests fast — they never trigger a network call or require an `OPENAI_API_KEY`.

## Two advisors, two levels

| Advisor | Question answered |
| --- | --- |
| `playbook/advisor.py` | "What vLLM flags should I use for this workload + GPU?" |
| `analyze/deployment_advisor.py` | "Which quantization/precision/config should I deploy, given quality requirements?" |

Neither replaces the other.
