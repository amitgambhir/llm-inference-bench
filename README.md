# llm-inference-bench

A lightweight, platform-agnostic benchmarking harness for LLM inference endpoints.
Works with **local vLLM, SGLang, Baseten, Red Hat OpenShift AI (RHOAI), AWS SageMaker,
Azure ML, GCP Vertex** — anything that exposes an OpenAI-compatible `/v1/completions`
interface.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [What this tool does](#2-what-this-tool-does)
3. [Quick start — no GPU needed](#3-quick-start--no-gpu-needed)
4. [Quick start — against a running endpoint](#4-quick-start--against-a-running-endpoint)
5. [Quality-aware benchmarking](#5-quality-aware-benchmarking)
6. [Components](#6-components)
   - 6.1 [`collect/run_bench.py` — the benchmark harness](#61-collectrunbenchpy--the-benchmark-harness)
   - 6.2 [`analyze/report.py` — Markdown report generator](#62-analyzereportpy--markdown-report-generator)
   - 6.3 [`playbook/advisor.py` — config recommendation engine](#63-playbookadvisorpy--config-recommendation-engine)
   - 6.4 [`evaluate/run_eval.py` — quality evaluator](#64-evaluaterunevalpy--quality-evaluator)
   - 6.5 [`analyze/deployment_advisor.py` — deployment decision engine](#65-analyzedeploymentadvisorpy--deployment-decision-engine)
   - 6.6 [`data/generate_synthetic.py` — reference dataset](#66-datageneratesyntheticpy--reference-dataset)
   - 6.7 [`workloads/*.yaml` — workload profiles](#67-workloadsyaml--workload-profiles)
   - 6.8 [`examples/` — platform-specific guides](#68-examples--platform-specific-guides)
   - 6.9 [`results/` — collected and reference data](#69-results--collected-and-reference-data)
7. [Key findings from the validation run](#7-key-findings-from-the-validation-run)
8. [Project structure](#8-project-structure)
9. [Data lifecycle and gitignore](#9-data-lifecycle-and-gitignore)
10. [Contributing](#10-contributing)
11. [License](#11-license)

---

## 1. The problem

Every team running an LLM in production eventually hits the same wall:

> *"Our model is slow / expensive / unreliable under load. What knob do we turn?"*

The vLLM and SGLang documentation list **dozens** of tuning flags — `max-num-seqs`,
`enable-chunked-prefill`, `enable-prefix-caching`, `tensor-parallel-size`, `kv-cache-fraction`, `swap-space`, and so on. Most public guidance is generic ("enable chunked
prefill for long contexts") and doesn't account for the specific **GPU**, **model
precision**, and **workload shape** in front of you.

The result: engineers either (a) cargo-cult a config from a blog post, (b) over-
provision hardware to compensate, or (c) spend weeks running ad-hoc benchmarks
with one-off scripts.

This project answers the question that every ML platform engineer and solutions
architect faces:

> **"For *my* workload and *my* hardware, what vLLM configuration actually matters
> — and how do I measure it?"**

It does so by combining three things:

1. A **measurement tool** that fires realistic, concurrent load at any
   OpenAI-compatible endpoint and captures TTFT, throughput, and failure rate.
2. An **analysis layer** that turns raw measurements into a comparison report.
3. A **recommendation engine** whose rules are anchored in real benchmark data
   (NVIDIA L4 + Llama 3.1 8B FP8, validated on Red Hat OpenShift AI) — not
   blog-post intuition.

---

## 2. What this tool does

- **Measures** time-to-first-token (p50/p95/p99), end-to-end latency, throughput
  (tokens/s and requests/s), and failure rate against any streaming
  `/v1/completions` endpoint.
- **Analyzes** results across runs and produces a deployable Markdown report that
  surfaces ISL impact, the `max-num-seqs` sweep, concurrency scaling, chunked
  prefill comparison, and prefix-cache wins.
- **Recommends** a concrete vLLM configuration (`--max-num-seqs`,
  `--enable-chunked-prefill`, `--enable-prefix-caching`, replica count) for a
  given workload + hardware combination via a small, transparent rule engine.
- **Evaluates quality** against real eval prompts (50–500 samples) and produces a
  quality sidecar JSON linked to the latency result. Then **recommends** the best
  deployment balancing latency, cost, and quality — not just the fastest one.
- **Stays platform-agnostic.** The core harness has no notion of "vLLM" or
  "Baseten" or "RHOAI" — it only knows OpenAI-compatible streaming. Platform
  specifics live under `examples/`.

---

## 3. Quick start — no GPU needed

The synthetic-data path exercises the whole pipeline in under a minute without
touching a GPU. Useful for demos, CI, and verifying the install.

```bash
# 1. Clone and enter the repo
git clone <this-repo>
cd llm-inference-bench

# 2. Generate the reference dataset (stdlib only, no install needed)
python data/generate_synthetic.py

# 3. Build the report
python analyze/report.py --output report.md

# 4. Get a recommendation
python playbook/advisor.py \
  --isl 2048 --latency-sla 700 --concurrency 20 \
  --scale mixed --gpu l4 --model-precision fp8
```

`generate_synthetic.py`, `report.py`, and `advisor.py` are **stdlib-only** — no
`pip install` needed. `aiohttp` is required only for the live benchmark in step 4
of the next section.

---

## 4. Quick start — against a running endpoint

```bash
pip install -r requirements.txt    # installs aiohttp

python collect/run_bench.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --isl 2048 --osl 128 \
  --concurrency 10 --duration 90 \
  --tag my_first_run

python analyze/report.py --output report.md
```

For authenticated endpoints (Baseten, RHOAI with auth enabled, cloud-managed
inference), add `--token $API_KEY`. The benchmark adds a `Bearer` header only
when `--token` is provided, so it works with both authenticated and open endpoints
from the same script.

See the [platform guides](#68-examples--platform-specific-guides) for full
walkthroughs against local vLLM, Baseten, and RHOAI.

---

## 5. Quality-aware benchmarking

Most inference benchmarks answer: *"Which configuration is fastest?"* This tool goes further:

> **"Is the faster deployment actually good enough — in quality and cost?"**

### Pipeline

```text
collect/run_bench.py      →  results/real/<tag>.json         (latency)
evaluate/run_eval.py      →  results/quality/<tag>.json      (quality sidecar)
analyze/deployment_advisor.py --tags t1 t2 t3               (recommendation)
```

The evaluation pipeline is **offline and separate** from the load test — it sends 50–500 representative prompts at low concurrency (5) so it does not warm the KV cache or interfere with a parallel load test.

### Example

```bash
# 1. Run the quality evaluator against a deployment
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/real/vllm_l4fp8_isl2k_c10.json \
  --dataset datasets/rag.jsonl \
  --evaluator deepeval \
  --eval-model gpt-4o \
  --cost-per-million-tokens 0.80

# 2. Compare deployments
python analyze/deployment_advisor.py \
  --tags vllm_a100fp16 vllm_l4fp8 vllm_l4int4 \
  --baseline vllm_a100fp16 \
  --quality-threshold 0.10
```

**Output:**

```text
=== Deployment Recommendation ===

Recommended: vllm_l4fp8

  Latency Improvement:  42.5%  (200ms → 115ms TTFT p50)
  Cost Reduction:       33.3%  ($1.20 → $0.80 per 1M tokens)
  Quality Delta:        -2.0%  (0.900 → 0.880)

Eliminated: vllm_l4int4 — quality drop 18.0% exceeds threshold (10.0%)

Tradeoff Table:
  Tag                    TTFT p50   Tok/s    Quality   Cost/1M    Status
  vllm_a100fp16           200ms      200    0.900      $1.20     baseline
  vllm_l4fp8              115ms      262    0.880      $0.80     RECOMMENDED
  vllm_l4int4              80ms      400    0.720      $0.50     eliminated
```

### Quality sources

| Evaluator | When to use | Requirement |
| --- | --- | --- |
| `deepeval` (default) | Automated CI runs, reproducible scoring | `pip install deepeval` + OpenAI API key for judge model |
| `llm-judge` | Any OpenAI-compatible judge endpoint | Same flags, no extra install |

### Datasets

Three built-in eval datasets under `datasets/`:

| File | Workload | Prompts |
| --- | --- | --- |
| `datasets/chat.jsonl` | Customer support / enterprise chat | 15 |
| `datasets/rag.jsonl` | RAG over claim documents | 15 |
| `datasets/long_context.jsonl` | Long-document analysis and summarization | 15 |

Each row: `{"schema_version": 1, "id": "...", "workload": "...", "prompt": "...", "expected": "..."}`.

Use `--dry-run` on either script to validate inputs without hitting endpoints or the judge model.

---

## 6. Components

The repo has five executable pieces (`run_bench`, `report`, `advisor`,
`generate_synthetic`, plus the example assets). They are intentionally
**decoupled** — the benchmark writes JSON, everything else reads it. This means
the GPU is only needed for data collection; analysis and recommendations run
anywhere.

### 6.1 `collect/run_bench.py` — the benchmark harness

The only component that needs a network connection to a model.

**What it does.** Spawns `--concurrency` async workers that each issue
streaming requests to `--endpoint` for `--duration` seconds. For every request
it captures:

- **TTFT** (time to first token) — measured as the wall-clock delta between
  request send and the first non-empty streamed token chunk. This is the
  metric users actually feel.
- **End-to-end latency** — time until `[DONE]` or stream close.
- **Tokens generated** — used to compute throughput.
- **Success/failure** — non-200 responses and exceptions are counted, not
  crashed on.

**Key arguments.**

| Flag | Purpose |
| --- | --- |
| `--endpoint` | OpenAI-compatible completions URL (default `http://localhost:8000/v1/completions`) |
| `--model` | Model name as served by the runtime (required) |
| `--isl` | Approximate input sequence length in tokens (default 512). Selects one of three realistic enterprise prompts (512/2048/4096) — *not* lorem ipsum |
| `--osl` | Max output tokens (default 128) |
| `--concurrency` | Number of concurrent virtual users (default 10) |
| `--duration` | Test duration in seconds (default 90) |
| `--tag` | Result filename tag, e.g. `vllm_isl2k_c10` (required) |
| `--token` | Bearer token for authenticated endpoints (optional) |
| `--runtime` | Metadata only: `vllm`, `sglang`, etc. (default `vllm`) |
| `--chunked-prefill` | Metadata only — records which config was active for this run |
| `--shared-prefix` | Prepends a fixed system prompt to every request, for prefix-cache testing |
| `--output-dir` | Where to write the JSON (default `./results/real`; supports `~/path`) |

**Output.** A single JSON file at `<output-dir>/<tag>.json` with the schema:

```json
{
  "meta": {
    "tag": "vllm_isl2k_c10",
    "runtime": "vllm",
    "model": "llama-3.1-8b",
    "gpu": {"name": "NVIDIA L4", "memory_mb": 23034, "util_pct": 0},
    "config": {"chunked_prefill": false, "tensor_parallel_size": 1, "shared_prefix": false},
    "workload": {"isl_approx": 2048, "osl_max": 128, "concurrency": 10, "duration_secs": 90},
    "synthetic": false,
    "timestamp": "2026-05-23T16:17:04+00:00"
  },
  "metrics": {
    "ttft_ms": {"p50": 115.1, "p90": 116.5, "p95": 132.8, "p99": 194.8, "mean": 110.4},
    "total_latency_ms": {"p50": 4966.6, "p95": 4987.1, "p99": 5012.5},
    "throughput_tokens_per_sec": 261.7,
    "throughput_req_per_sec": 2.04,
    "total_requests": 190,
    "successful_requests": 184,
    "failed_requests": 6
  }
}
```

**Design notes.**

- All hot-path globals (`ENDPOINT`, `MODEL`, `TOKEN`, `USE_SHARED_PREFIX`) are
  declared at module level and assigned once at the top of `main()` so worker
  coroutines avoid argument-passing overhead.
- The auth header is added **only** when `--token` is non-empty, so the same
  script works against open localhost endpoints and authenticated cloud endpoints.
- GPU info comes from `nvidia-smi`; absence falls back to `"unknown"` rather
  than crashing — important for running the harness from a bastion that does
  not have GPUs.
- The three built-in prompts (insurance support / claim review / portfolio
  review) simulate realistic enterprise workloads. Synthetic lorem ipsum
  changes prefill cost characteristics and produces misleading numbers.

### 6.2 `analyze/report.py` — Markdown report generator

Reads every JSON file under `results/real/` and `results/synthetic/` and emits
a single Markdown report. **Real measurements override synthetic rows that
share the same `tag`** — so once you start collecting real data, the synthetic
reference quietly steps out of the way.

**Sections produced.**

1. **Header** — runtimes, GPUs, models, run counts, timestamp range.
2. **ISL impact table** — TTFT and throughput as a function of input length at
   low concurrency.
3. **Chunked prefill comparison** — paired off/on rows when both exist, with a
   callout explaining the L4-FP8 zero-benefit finding when relevant.
4. **`max-num-seqs` sweep** — the headline 172x finding.
5. **Concurrency vs throughput** — saturation curve at fixed ISL.
6. **Prefix caching comparison** — on vs off.
7. **Deployment recommendations table** — chat / RAG / long-context profiles.
8. **Methodological notes** — what's measured, what's an upper bound, the
   network-floor caveat.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--output` | Output path (default `-` for stdout) |
| `--real-only` | Skip `results/synthetic/` entirely — useful once you have enough real coverage |

**Stdlib only.** No dependencies. Runs anywhere Python 3.9+ runs.

### 6.3 `playbook/advisor.py` — config recommendation engine

A small CLI rule engine that converts a workload description into a concrete
vLLM configuration. **Every rule is grounded in real benchmark data**, with
the source cited inline in the rationale string.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--isl` | Input sequence length (tokens) |
| `--latency-sla` | TTFT p95 SLA in milliseconds |
| `--concurrency` | Expected concurrent active users |
| `--scale` | `realtime` / `mixed` / `batch` — controls capacity headroom |
| `--gpu` | `l4` / `l40s` / `a100` / `h100` (default `l4`) |
| `--model-precision` | `fp8` / `fp16` (default `fp8`) |
| `--interactive` | Prompt for inputs interactively |
| `--json` | Emit machine-readable JSON instead of formatted text |

**Rules encoded.**

- **`max-num-seqs` = `max(concurrency × 2, 64)`, capped at 256.** Rationale:
  L4 FP8 at c=50 went from TTFT p50=24,554ms (mns=8) to 143ms (mns=128) — a
  172x improvement. This is the single most impactful parameter.
- **Chunked prefill on L4/L40S FP8: disabled.** FP8 makes prefill fast enough
  that monopolization isn't the bottleneck this flag targets. Real test at
  ISL=4096, c=50 showed zero benefit.
- **Chunked prefill on A100/H100 FP16 with ISL>1024: enabled.** Literature
  shows 3–5x p95 improvement where prefill dominates.
- **Prefix caching: always enabled.** No downside; external benchmarks may
  not show the latency win due to network floor, but the GPU-level savings
  are real.
- **Replica count:** anchor 5.5 req/s sustainable per L4 FP8 replica at
  ISL=2048; scaled by `sqrt(2048/isl)`; headroom 40% for realtime, 25% for
  mixed, 10% for batch.

**Warnings emitted.**

- `concurrency > max-num-seqs` — requests will queue, TTFT will spike.
- `gpu=l4` + `precision=fp16` — VRAM pressure for 8B+ models.
- `scale=realtime` + `isl>4096` — prefill alone may exceed the latency budget.

### 6.4 `evaluate/run_eval.py` — quality evaluator

Sends eval dataset prompts to an inference endpoint at low concurrency (5 workers), scores responses with DeepEval or an LLM judge, and writes a quality sidecar JSON to `results/quality/<tag>.json`.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--endpoint` | Inference endpoint to evaluate (required) |
| `--model` | Model name as served (required) |
| `--latency-result` | Path to the latency JSON this eval is paired with (required — sets the `latency_tag` backlink) |
| `--dataset` | Path to a JSONL eval dataset (required) |
| `--evaluator` | `deepeval` (default) or `llm-judge` |
| `--eval-model` | Judge model for DeepEval or LLM-judge (default: `gpt-4o`) |
| `--eval-endpoint` | Judge endpoint (default: `https://api.openai.com/v1`) |
| `--token` | Bearer token for the inference endpoint |
| `--eval-token` | Bearer token for the judge endpoint |
| `--cost-per-million-tokens` | Optional. Recorded in the quality sidecar for cost comparison |
| `--output-dir` | Where to write the quality sidecar (default: `results/quality`) |
| `--dry-run` | Validate inputs and print what would run, then exit |

**DeepEval metrics by workload.**

| Workload | Metrics |
| --- | --- |
| `chat`, `long_context` | `AnswerRelevancyMetric`, `GEval(correctness)` |
| `rag` (no `contexts` field) | same as above |
| `rag` (with `contexts` field) | above + `FaithfulnessMetric`, `HallucinationMetric` |

**Requires:** `pip install deepeval` (already in `requirements.txt`). When `--eval-endpoint` or `--eval-token` are provided, the process-level `OPENAI_API_KEY` / `OPENAI_BASE_URL` environment variables are set before DeepEval runs — safe for the CLI, but be aware of this side effect if calling `run_deepeval` as a library.

**Dataset validation.** `load_dataset` requires `schema_version: 1` in every row — any other value is a hard error. Valid workloads: `"chat"`, `"rag"`, `"long_context"`.

### 6.5 `analyze/deployment_advisor.py` — deployment decision engine

Answers: **"Given my quality requirements and cost constraints, which deployment should I choose?"**

Loads latency results and quality sidecars for each tag, computes relative deltas vs a baseline, eliminates deployments that fall below the quality threshold, and recommends the best surviving option by latency improvement.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--tags` | Deployment tags to compare (space-separated, required) |
| `--baseline` | Tag to treat as the reference (required) |
| `--quality-threshold` | Max acceptable quality drop vs baseline (default: `0.10` = 10%) |
| `--output` | `markdown` (default, terminal-friendly) or `json` |
| `--latency-dirs` | Override latency search dirs (default: `results/synthetic results/real`) |
| `--quality-dir` | Override quality sidecar dir (default: `results/quality`) |
| `--dry-run` | Load all tags and print a summary, then exit |

**Cost model.** Uses `--cost-per-million-tokens` from the quality sidecar when available on both profiles. Falls back to throughput ratio as a proxy. Shows "N/A" when neither is available.

**Hard errors.** The advisor stops immediately (with a clear message) on:

- Tag not found in any latency directory
- `latency_tag` in quality sidecar doesn't match the loaded tag — stale sidecar can silently corrupt a recommendation
- `meta.model` in quality sidecar differs from latency result model
- Profiles being compared carry quality scores from different eval datasets — results are not comparable

**Relation to `playbook/advisor.py`.** The playbook advisor answers "what vLLM flags should I use?" The deployment advisor answers "which quantization / precision / configuration should I deploy?" Two advisors, two levels of the stack — neither replaces the other.

### 6.6 `data/generate_synthetic.py` — reference dataset

Generates ~23 JSON files under `results/synthetic/` containing scenarios that
are too expensive or impractical to measure on every workstation: A100/H100
FP16 comparisons, SGLang/vLLM throughput differences, ISL interpolation
between measured points, the full `max-num-seqs` sweep, and so on.

**All synthetic rows are tagged `synthetic: true`.** The values are
extrapolated conservatively from six anchor measurements taken on the real
L4/FP8 validation run (documented in [BENCHMARK_FINDINGS.md](BENCHMARK_FINDINGS.md)):

| Anchor | TTFT p50 | TTFT p95 | Throughput |
| --- | ---: | ---: | ---: |
| ISL=512, c=10 | 75ms | 81ms | 262 tok/s |
| ISL=2048, c=10 | 115ms | 133ms | 262 tok/s |
| ISL=4096, c=50 | 134ms | 335ms | 641 tok/s |
| `mns=8`, c=50 | 24,554ms | — | 206 tok/s |
| `mns=32`, c=50 | 7,787ms | — | 502 tok/s |
| `mns=128`, c=50 | 143ms | — | 714 tok/s |

When real data overrides a synthetic row, the report quietly upgrades. This
gives newcomers a complete-looking dataset on day 1 and a path to displace
synthetics with real measurements over time.

### 6.7 `workloads/*.yaml` — workload profiles

Three YAML workload profiles you can pass directly to `advisor.py` (or use as
config inputs in your own tooling):

| File | ISL | OSL | Concurrency | SLA | Scale |
| --- | ---: | ---: | ---: | ---: | --- |
| [`chat.yaml`](workloads/chat.yaml) | 512 | 128 | 20 | 300ms | realtime |
| [`rag.yaml`](workloads/rag.yaml) | 2048 | 256 | 30 | 700ms | mixed |
| [`long_context.yaml`](workloads/long_context.yaml) | 4096 | 512 | 50 | 2000ms | batch |

Each file also records the baseline L4/FP8 measurement for that profile so
you can sanity-check your own numbers against the validation run.

### 6.8 `examples/` — platform-specific guides

The benchmark harness is generic; deployment platforms aren't. These guides
get you from "I have an account" to "I have a benchmark JSON" on each
platform:

- **[`examples/local-vllm/QUICKSTART.md`](examples/local-vllm/QUICKSTART.md)** —
  install vLLM, serve a model, benchmark it. Covers Docker, SGLang as a drop-in
  alternative, and GPU memory requirements by model size.
- **[`examples/baseten/QUICKSTART.md`](examples/baseten/QUICKSTART.md)** —
  benchmark a Baseten-deployed model. Covers endpoint URL shape, API keys,
  cold-start warming, and how to A/B latency-optimized vs throughput-optimized
  deployments.
- **[`examples/rhoai/RUNBOOK.md`](examples/rhoai/RUNBOOK.md)** — the full
  runbook used to collect every "real" data point in this repo. Includes
  `ServingRuntime` and `InferenceService` manifests, route exposure,
  parameter-patching commands, and troubleshooting for the issues encountered
  (orphaned `LLMInferenceService` CRDs, route port-forwarding gotchas, token
  expiry).
  - [`examples/rhoai/serving_runtime.yaml`](examples/rhoai/serving_runtime.yaml) — the RHAIIS `ServingRuntime`
  - [`examples/rhoai/isvc.yaml`](examples/rhoai/isvc.yaml) — the `InferenceService` deploying Llama 3.1 8B FP8 from an OCI registry

### 6.9 `results/` — collected and reference data

Two siblings:

- `results/real/` — populated by `run_bench.py`. Gitignored by default (see
  [section 9](#9-data-lifecycle-and-gitignore)). Each file represents one
  benchmark run.
- `results/synthetic/` — populated by `generate_synthetic.py`. Committed to
  the repo so a fresh clone has working data immediately.

`report.py` reads both directories. When the same `tag` appears in both, the
real measurement wins.

---

## 7. Key findings from the validation run

Validated on **NVIDIA L4 + Llama 3.1 8B FP8** via Red Hat OpenShift AI. See
[BENCHMARK_FINDINGS.md](BENCHMARK_FINDINGS.md) for the full study including
infrastructure setup, experiment design, and per-run numbers.

| Parameter | Finding |
| --- | --- |
| `max-num-seqs` 8 → 128 | **172x** TTFT improvement at c=50 (24,554ms → 143ms) |
| Chunked prefill on L4 FP8 | No measurable benefit — FP8 eliminates the problem it solves |
| Concurrency 10 → 50 | 2.4x throughput (262 → 641 tok/s) |
| External vs internal benchmarking | ~15–30ms network floor masks sub-30ms GPU optimizations |

> **These findings are hardware-specific (NVIDIA L4, FP8).** Results will
> differ on A100/H100 with FP16 — chunked prefill in particular tends to help
> there. The tool is platform-agnostic; the *interpretation* is hardware-
> aware. `advisor.py`'s rules encode this distinction explicitly.

---

## 8. Project structure

```text
llm-inference-bench/
├── README.md                       # this file
├── BENCHMARK_FINDINGS.md           # full L4/FP8 validation study
├── requirements.txt                # aiohttp only — used by run_bench.py
├── .gitignore
│
├── collect/
│   └── run_bench.py                # async benchmark; OpenAI-compatible; streaming TTFT
│
├── evaluate/
│   └── run_eval.py                 # quality evaluator; DeepEval + LLM-judge; low concurrency
│
├── datasets/
│   ├── chat.jsonl                  # 15 enterprise chat eval prompts
│   ├── rag.jsonl                   # 15 RAG eval prompts (claim documents)
│   └── long_context.jsonl          # 15 long-document analysis prompts
│
├── analyze/
│   ├── report.py                   # JSON → Markdown report (stdlib only)
│   └── deployment_advisor.py       # latency + quality + cost → deployment recommendation
│
├── playbook/
│   └── advisor.py                  # workload + hardware → vLLM config (stdlib only)
│
├── data/
│   └── generate_synthetic.py       # reference dataset from real L4/FP8 anchors
│
├── workloads/
│   ├── chat.yaml                   # realtime, ISL=512, SLA=300ms
│   ├── rag.yaml                    # mixed, ISL=2048, SLA=700ms
│   └── long_context.yaml           # batch, ISL=4096, SLA=2000ms
│
├── results/
│   ├── real/                       # populated by run_bench.py (gitignored)
│   │   └── .gitkeep
│   ├── synthetic/                  # populated by generate_synthetic.py (committed)
│   └── quality/                    # populated by run_eval.py (quality sidecars)
│       └── .gitkeep
│
└── examples/
    ├── local-vllm/
    │   └── QUICKSTART.md           # local vLLM / Docker / SGLang in 5 minutes
    ├── baseten/
    │   └── QUICKSTART.md           # benchmark a Baseten-deployed model
    └── rhoai/
        ├── serving_runtime.yaml    # RHAIIS ServingRuntime (vLLM 3.2.4)
        ├── isvc.yaml               # InferenceService (Llama 3.1 8B FP8)
        └── RUNBOOK.md              # full OpenShift AI walkthrough + troubleshooting
```

---

## 9. Data lifecycle and gitignore

`results/real/*.json` is **gitignored by default** to keep the repo small and
avoid leaking sensitive endpoint identifiers in result metadata. The
`results/real/.gitkeep` placeholder keeps the directory itself present on a
fresh clone.

If you want to commit your benchmark data alongside the analysis (recommended
for reproducible postmortems and team-shared experiments), remove the
`results/real/*.json` line from `.gitignore`.

`results/synthetic/*.json` is committed so a fresh clone has a working dataset
out of the box.

---

## 10. Contributing

PRs welcome. Two principles:

1. **Real data beats synthetic.** When adding or changing a rule in
   `advisor.py`, anchor it to a measurement (in this repo or in published
   literature) and cite the source in the rationale string. Vibe-based rules
   get rejected.
2. **Stay platform-agnostic in `collect/`, `analyze/`, and `playbook/`.**
   Anything specific to a deployment platform (RHOAI, Baseten, SageMaker,
   Vertex, etc.) goes under `examples/`. The core tool must never grow a
   conditional on "which cloud are we on".
3. **Quality data is not optional for production decisions.** When comparing
   deployments, run `evaluate/run_eval.py` before `deployment_advisor.py`. A
   recommendation made without quality data ranks by latency only — acceptable
   for a quick filter, not for a ship decision.

Python 3.9+ compatible. `collect/run_bench.py` uses `aiohttp`; everything
else is stdlib-only.

---

## 11. License

Apache 2.0.
