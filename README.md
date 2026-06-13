# llm-inference-bench

> **From a business workload spec to a production-ready LLM inference configuration — without guessing.**

Most teams decide how many GPUs to buy, which quantization to use, and what vLLM flags to set by copying a blog post or over-provisioning to buy safety margin. This tool replaces that guesswork with a structured, measurement-driven workflow:

1. **Describe your workload** (requests/day, token lengths, latency SLO) — get a replica estimate and cost envelope *before* you touch any hardware.
2. **Run the generated benchmark plan** against your actual GPU — real numbers replace the estimate, confidence upgrades automatically.
3. **Compare deployments** on latency, cost, *and* quality together — not just the fastest option, but the best option that meets your quality bar.

Ships as a **web app** (Next.js + FastAPI, deployable on Vercel + Render in minutes) and as a **CLI** for automation pipelines.

---

## Key outputs

| Stage | What you get |
| --- | --- |
| Before hardware | Replica count, cost envelope (on-demand + reserved), confidence level (HIGH/MEDIUM/LOW), calibrated benchmark plan |
| After benchmarking | Validated latency profiles (TTFT p50/p95/p99, throughput, failure rate), upgraded confidence |
| Deployment decision | Ranked recommendation across latency, cost, and quality — with eliminated options explained |

---

## Important context

This is a **comparative inference evaluation framework**, not a hardware microbenchmarking tool.

Results are:

- Valid within a defined and reproducible runtime environment
- Intended for deployment decisioning, not absolute performance claims
- Dependent on underlying infrastructure (GPU, runtime engine, and kernel optimizations)

---

## Table of contents

1. [The problem](#1-the-problem)
2. [What this tool does](#2-what-this-tool-does)
3. [Quick start - no GPU needed](#3-quick-start--no-gpu-needed)
4. [Quick start - against a running endpoint](#4-quick-start--against-a-running-endpoint)
5. [Quality-aware benchmarking](#5-quality-aware-benchmarking)
6. [Capacity Planner](#6-capacity-planner)
   - 6.1 [Quick start – capacity planner CLI](#61-quick-start--capacity-planner-cli)
   - 6.2 [Quick start – API + UI](#62-quick-start--api--ui)
   - 6.3 [`planner/` modules](#63-planner-modules)
   - 6.4 [`api/` – REST endpoints](#64-api--rest-endpoints)
   - 6.5 [`catalog/` – GPU and model catalog](#65-catalog--gpu-and-model-catalog)
   - 6.6 [`ui/` – four-screen web interface](#66-ui--four-screen-web-interface)
7. [Components](#7-components)
   - 7.1 [`collect/run_bench.py` - the benchmark harness](#71-collectrunbenchpy--the-benchmark-harness)
   - 7.2 [`analyze/report.py` - Markdown report generator](#72-analyzereportpy--markdown-report-generator)
   - 7.3 [`playbook/advisor.py` - config recommendation engine](#73-playbookadvisorpy--config-recommendation-engine)
   - 7.4 [`evaluate/run_eval.py` - quality evaluator](#74-evaluaterunevalpy--quality-evaluator)
   - 7.5 [`analyze/deployment_advisor.py` - deployment decision engine](#75-analyzedeploymentadvisorpy--deployment-decision-engine)
   - 7.6 [`data/generate_synthetic.py` - reference dataset](#76-datageneratesyntheticpy--reference-dataset)
   - 7.7 [`workloads/*.yaml` - workload profiles](#77-workloadsyaml--workload-profiles)
   - 7.8 [`examples/` - platform-specific guides](#78-examples--platform-specific-guides)
   - 7.9 [`results/` - collected and reference data](#79-results--collected-and-reference-data)
8. [Key findings from the validation run](#8-key-findings-from-the-validation-run)
9. [Project structure](#9-project-structure)
10. [Data lifecycle and gitignore](#10-data-lifecycle-and-gitignore)
11. [Contributing](#11-contributing)
12. [License](#12-license)

---

## 1. The problem

LLM inference deployments fail in three distinct ways, at three distinct stages:

**Before you have hardware:**

> *"We need to provision GPUs for our new LLM feature. How many do we need?"*

Nobody has a good answer. The capacity estimate is usually a gut feeling, a competitor's blog post, or a spreadsheet built on single-request latency numbers that have nothing to do with concurrent production traffic. Teams either under-provision (SLO misses on day one) or over-provision by 2–3× to buy safety margin — wasting tens of thousands of dollars per month.

**When you have hardware but no config:**

> *"Which of these 40 vLLM flags actually matter for my model and GPU?"*

The documentation lists `max-num-seqs`, `enable-chunked-prefill`, `enable-prefix-caching`, `tensor-parallel-size`, `kv-cache-fraction`, and more. Generic guidance ("enable chunked prefill for long contexts") ignores the specific interaction between GPU memory bandwidth, model precision, and workload shape. Real data — L4 FP8, `max-num-seqs` 8 → 128 — shows a **172× TTFT improvement** that no blog post would have predicted.

**When you're choosing between deployment options:**

> *"FP8 is 40% faster and 30% cheaper than FP16 — but is the quality still acceptable?"*

Most teams never check. They pick the fastest or cheapest option and ship it. Quality degradation from aggressive quantization shows up later, in user complaints, not in a benchmark report.

---

This tool solves all three. It gives every ML platform engineer and solutions architect a structured, measurement-driven path from *"we need to serve this model"* to *"this is our validated production config"*:

1. A **capacity planner** that estimates replicas and cost from a business workload spec — no GPU required.
2. A **benchmark harness** that fires realistic concurrent load at any OpenAI-compatible endpoint and produces a structured, reproducible result.
3. A **deployment advisor** whose recommendations are anchored in real measured data (NVIDIA L4 + Llama 3.1 8B FP8, validated on Red Hat OpenShift AI) and account for quality alongside latency and cost.

---

## 2. What this tool does

**Plan** — before you have a GPU:

- Describe your workload: requests/day, input/output token lengths, TTFT SLO, traffic class
- Get back: recommended replica count, low/high range, binding constraint (`compute`, `bandwidth`, or `kv_budget`), cost envelope (on-demand + 1-yr reserved), and a confidence level with an explicit uncertainty band (HIGH ±10%, MEDIUM ±25%, LOW ±50%)
- Get a prioritized benchmark plan — ordered CLI commands that collapse the biggest uncertainty first

**Measure** — against a real endpoint:

- Fires realistic concurrent load (`--concurrency` async workers for `--duration` seconds) at any streaming `/v1/completions` endpoint
- Captures TTFT (p50/p95/p99), end-to-end latency, throughput (tokens/s, req/s), and failure rate
- Embeds the full environment contract in every result file (GPU, model, precision, workload shape) so runs are reproducible and comparable
- Runs a separate quality evaluation pass (50–500 eval prompts at low concurrency) to score answer relevancy, correctness, faithfulness, and hallucination rate

**Decide** — across deployment options:

- Compares configurations on latency, cost, and quality simultaneously
- Eliminates options that fall below a quality threshold — not just the slowest ones
- Recommends a concrete vLLM config (`--max-num-seqs`, `--enable-chunked-prefill`, `--enable-prefix-caching`, replica count) grounded in real benchmark data
- Produces a Markdown report with a validation badge (`estimate_only` → `partially_validated` → `validated_by_benchmark`) that reflects how much real GPU evidence backs the recommendation

**Deploy anywhere** — the harness has no coupling to any platform. It speaks OpenAI-compatible streaming. Platform specifics (vLLM local, Baseten, RHOAI) live under `examples/` and the hosted web app works with any endpoint URL.

---

## 3. Quick start - no GPU needed

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

`generate_synthetic.py`, `report.py`, and `advisor.py` are **stdlib-only** - no
`pip install` needed. `aiohttp` is required only for the live benchmark in step 4
of the next section.

---

## 4. Quick start - against a running endpoint

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

For authenticated endpoints, pass one of:

```bash
--token $API_KEY          # Bearer token (Baseten, RHOAI, cloud-managed)
--basic-auth user:pass    # HTTP Basic Auth (self-hosted with nginx proxy, etc.)
```

Both flags are optional; omit both for open endpoints.

See the [platform guides](#78-examples--platform-specific-guides) for full
walkthroughs against local vLLM, Baseten, and RHOAI.

---

## 5. Quality-aware benchmarking

Most inference benchmarks answer: *"Which configuration is fastest?"* This tool goes further:

> **"Is the faster deployment actually good enough - in quality and cost?"**

### Pipeline

```text
collect/run_bench.py      →  results/real/<tag>.json         (latency)
evaluate/run_eval.py      →  results/quality/<tag>.json      (quality sidecar)
analyze/deployment_advisor.py --tags t1 t2 t3               (recommendation)
```

The evaluation pipeline is **offline and separate** from the load test - it sends 50–500 representative prompts at low concurrency (5) so it does not warm the KV cache or interfere with a parallel load test.

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

Eliminated: vllm_l4int4 - quality drop 18.0% exceeds threshold (10.0%)

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

## 6. Capacity Planner

Size a deployment *before* you have a GPU. Given a business workload description (requests/day, ISL, OSL, TTFT SLO, traffic class), the planner applies a roofline model to estimate replica count, cost envelope, and confidence level, then generates an ordered benchmark plan to validate the estimate. When benchmark runs complete, `ingest_anchor.py` calibrates the model with real GPU data, upgrading confidence from LOW → MEDIUM → HIGH.

### 6.1 Quick start – capacity planner CLI

```bash
pip install -r requirements.txt

python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --requests-per-day 50000 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime
```

Output: recommended replicas, confidence level (HIGH / MEDIUM / LOW), TTFT estimate (ms), max concurrent sequences (KV budget), and the binding constraint (`compute`, `bandwidth`, or `kv_budget`).

To compare multiple GPU/dtype configurations head-to-head:

```bash
python planner/compare.py --configs configs.json
```

Output: cheapest configuration, safest (highest confidence), best latency — with tradeoff notes (e.g., H200 vs H100 KV budget delta).

### 6.2 Quick start – API + UI

```bash
# Terminal 1 – start the API
uvicorn api.main:app --reload

# Terminal 2 – start the UI
cd ui && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The four-screen UI guides you through the full planning workflow.

| Screen | Route | Purpose |
| --- | --- | --- |
| Scenario Builder | `/` | Workload description + GPU/model/dtype selection; supports catalog models and custom model spec |
| Estimate | `/estimate` | Replica range chart (low/recommended/high), confidence badge, binding constraint, warnings |
| Benchmark Plan | `/benchmark-plan` | Prioritized test matrix with copy-ready `run_bench.py` commands and rationale |
| Report | `/report` | Recommendation summary, validation status badge, export to Markdown |

### 6.3 `planner/` modules

| Module | Role |
| --- | --- |
| `planner/capacity.py` | Roofline model — separate prefill (compute-bound) and decode (bandwidth-bound) phases; outputs replicas, TTFT estimate, KV budget |
| `planner/cost.py` | On-demand and 1-yr reserved cost envelope; reads pricing from `catalog/pricing.yaml` |
| `planner/benchmark_plan.py` | Ordered test matrix — ISL sweep, concurrency sweep, precision comparison, KV cache validation |
| `planner/confidence.py` | Three-tier rubric: HIGH (±10%), MEDIUM (±25%), LOW (±50%); `geometry_source="estimated"` downgrades one level |
| `planner/ingest_anchor.py` | Reads a completed benchmark JSON and writes a calibration anchor to `catalog/anchors.yaml` |
| `planner/compare.py` | Multi-config comparison: cheapest (cost_day_usd or replica proxy), safest (confidence rank × band), best latency |
| `planner/report.py` | Markdown report with mode badge (`estimate_only` / `partially_validated` / `validated_by_benchmark`) |

### 6.4 `api/` – REST endpoints

A FastAPI application with SQLite persistence (Postgres-ready). Start with `uvicorn api.main:app`. Uses `create_app()` factory pattern for test isolation — tests inject a `tmp_path` SQLite URL, not the production database.

| Endpoint | What it does |
| --- | --- |
| `POST /scenarios` | Create a new planning scenario |
| `GET /scenarios/{id}` | Fetch a scenario |
| `POST /scenarios/{id}/estimate` | Run the roofline estimator; stores result |
| `GET /scenarios/{id}/benchmark-plan` | Generate the ordered benchmark plan |
| `POST /benchmarks/run` | Enqueue a benchmark run as a background task |
| `GET /benchmarks/{run_id}` | Poll job status (`queued` / `running` / `done` / `failed`) |
| `POST /ingest/{run_id}` | Ingest a completed run into the anchor catalog |
| `GET /scenarios/{id}/recommendation` | Final recommendation; `mode` escalates automatically as benchmark runs complete |
| `GET /scenarios/{id}/report` | Fetch the Markdown report |

### 6.5 `catalog/` – GPU and model catalog

YAML files under `catalog/` contain the hardware and pricing data the planner reads at runtime. No hardcoded specs in code.

| File | Contents |
| --- | --- |
| `catalog/gpus.yaml` | Peak FLOPS, memory bandwidth, VRAM, and MFU estimate for each GPU SKU |
| `catalog/models.yaml` | Parameter count, hidden dim, layer count, KV head count per model |
| `catalog/pricing.yaml` | On-demand and 1-yr reserved cost per GPU-hour by provider/region |
| `catalog/anchors.yaml` | Measured throughput anchors written by `ingest_anchor.py` |

Five GPU SKUs shipped: `h100_sxm`, `h200_sxm`, `a100_80gb_sxm`, `l40s`, `l4`.

### 6.6 `ui/` – four-screen web interface

Next.js 14 (App Router) + Tailwind CSS + Recharts. Rewrites `/api/*` to the FastAPI server at `http://localhost:8000`.

Key components:

| Component | Purpose |
| --- | --- |
| `ReplicaRangeChart` | Recharts bar chart with confidence-colored bars (low/recommended/high) and a reference line at the recommended value |
| `ConfidenceBadge` | Shows level + band percentage, color-coded green/yellow/red |
| `ModeBadge` | Validation status — yellow (`estimate_only`), blue (`partially_validated`), green (`validated_by_benchmark`) |
| `CopyButton` | Clipboard copy with 1.5 s "✓ Copied" confirmation |

---

## 7. Components

The benchmark and analysis layer has nine components. They are intentionally
**decoupled** - each writes JSON, everything downstream reads it. The GPU is
only needed for data collection; analysis and recommendations run anywhere.

### 7.1 `collect/run_bench.py` - the benchmark harness

The only component that needs a network connection to a model.

**What it does.** Spawns `--concurrency` async workers that each issue
streaming requests to `--endpoint` for `--duration` seconds. For every request
it captures:

- **TTFT** (time to first token) - measured as the wall-clock delta between
  request send and the first non-empty streamed token chunk. This is the
  metric users actually feel.
- **End-to-end latency** - time until `[DONE]` or stream close.
- **Tokens generated** - used to compute throughput.
- **Success/failure** - non-200 responses and exceptions are counted, not
  crashed on.

**Key arguments.**

| Flag | Purpose |
| --- | --- |
| `--endpoint` | OpenAI-compatible completions URL (default `http://localhost:8000/v1/completions`) |
| `--model` | Model name as served by the runtime (required) |
| `--isl` | Approximate input sequence length in tokens (default 512). Selects one of three realistic enterprise prompts (512/2048/4096) - *not* lorem ipsum |
| `--osl` | Max output tokens (default 128) |
| `--concurrency` | Number of concurrent virtual users (default 10) |
| `--duration` | Test duration in seconds (default 90) |
| `--tag` | Result filename tag, e.g. `vllm_isl2k_c10` (required) |
| `--token` | Bearer token for authenticated endpoints (optional) |
| `--runtime` | Metadata only: `vllm`, `sglang`, etc. (default `vllm`) |
| `--chunked-prefill` | Metadata only - records which config was active for this run |
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

**Environment contract.** The `meta` block is the full environment contract for each run — it records GPU hardware (`gpu.name`, `gpu.memory_mb`), runtime and precision (`runtime`, `model`), serving config (`config.chunked_prefill`, `config.tensor_parallel_size`), and workload shape (`workload.isl_approx`, `workload.osl_max`, `workload.concurrency`). Results are only comparable across runs with the same or explicitly documented differences in these fields. No separate environment manifest is needed — the contract is embedded in every result file.

**Design notes.**

- All hot-path globals (`ENDPOINT`, `MODEL`, `TOKEN`, `USE_SHARED_PREFIX`) are
  declared at module level and assigned once at the top of `main()` so worker
  coroutines avoid argument-passing overhead.
- The auth header is added **only** when `--token` is non-empty, so the same
  script works against open localhost endpoints and authenticated cloud endpoints.
- GPU info comes from `nvidia-smi`; absence falls back to `"unknown"` rather
  than crashing - important for running the harness from a bastion that does
  not have GPUs.
- The three built-in prompts (insurance support / claim review / portfolio
  review) simulate realistic enterprise workloads. Synthetic lorem ipsum
  changes prefill cost characteristics and produces misleading numbers.

### 7.2 `analyze/report.py` - Markdown report generator

Reads every JSON file under `results/real/` and `results/synthetic/` and emits
a single Markdown report. **Real measurements override synthetic rows that
share the same `tag`** - so once you start collecting real data, the synthetic
reference quietly steps out of the way.

**Sections produced.**

1. **Header** - runtimes, GPUs, models, run counts, timestamp range.
2. **ISL impact table** - TTFT and throughput as a function of input length at
   low concurrency.
3. **Chunked prefill comparison** - paired off/on rows when both exist, with a
   callout explaining the L4-FP8 zero-benefit finding when relevant.
4. **`max-num-seqs` sweep** - the headline 172x finding.
5. **Concurrency vs throughput** - saturation curve at fixed ISL.
6. **Prefix caching comparison** - on vs off.
7. **Deployment recommendations table** - chat / RAG / long-context profiles.
8. **Methodological notes** - what's measured, what's an upper bound, the
   network-floor caveat.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--output` | Output path (default `-` for stdout) |
| `--real-only` | Skip `results/synthetic/` entirely - useful once you have enough real coverage |

**Stdlib only.** No dependencies. Runs anywhere Python 3.9+ runs.

### 7.3 `playbook/advisor.py` - config recommendation engine

A small CLI rule engine that converts a workload description into a concrete
vLLM configuration. **Every rule is grounded in real benchmark data**, with
the source cited inline in the rationale string.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--isl` | Input sequence length (tokens) |
| `--latency-sla` | TTFT p95 SLA in milliseconds |
| `--concurrency` | Expected concurrent active users |
| `--scale` | `realtime` / `mixed` / `batch` - controls capacity headroom |
| `--gpu` | `l4` / `l40s` / `a100` / `h100` (default `l4`) |
| `--model-precision` | `fp8` / `fp16` (default `fp8`) |
| `--interactive` | Prompt for inputs interactively |
| `--json` | Emit machine-readable JSON instead of formatted text |

**Rules encoded.**

- **`max-num-seqs` = `max(concurrency × 2, 64)`, capped at 256.** Rationale:
  L4 FP8 at c=50 went from TTFT p50=24,554ms (mns=8) to 143ms (mns=128) - a
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

- `concurrency > max-num-seqs` - requests will queue, TTFT will spike.
- `gpu=l4` + `precision=fp16` - VRAM pressure for 8B+ models.
- `scale=realtime` + `isl>4096` - prefill alone may exceed the latency budget.

### 7.4 `evaluate/run_eval.py` - quality evaluator

Sends eval dataset prompts to an inference endpoint at low concurrency (5 workers), scores responses with DeepEval or an LLM judge, and writes a quality sidecar JSON to `results/quality/<tag>.json`.

**Arguments.**

| Flag | Purpose |
| --- | --- |
| `--endpoint` | Inference endpoint to evaluate (required) |
| `--model` | Model name as served (required) |
| `--latency-result` | Path to the latency JSON this eval is paired with (required - sets the `latency_tag` backlink) |
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

**Requires:** `pip install deepeval` (already in `requirements.txt`). When `--eval-endpoint` or `--eval-token` are provided, the process-level `OPENAI_API_KEY` / `OPENAI_BASE_URL` environment variables are set before DeepEval runs - safe for the CLI, but be aware of this side effect if calling `run_deepeval` as a library.

**Dataset validation.** `load_dataset` requires `schema_version: 1` in every row - any other value is a hard error. Valid workloads: `"chat"`, `"rag"`, `"long_context"`.

### 7.5 `analyze/deployment_advisor.py` - deployment decision engine

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

**Normalized profile schema.** `load_deployment` merges the latency JSON and quality sidecar into a single in-memory structure used by all downstream functions:

```json
{
  "tag": "vllm_fp8",
  "model": "llama-3.1-8b",
  "latency": {
    "ttft_ms_p50": 115,
    "ttft_ms_p95": 133,
    "throughput_tokens_per_sec": 262
  },
  "quality": {
    "overall_score": 0.93,
    "metrics": {"answer_relevancy": 0.94, "correctness": 0.92}
  },
  "num_samples": 15,
  "cost": {
    "per_million_tokens": 0.80,
    "throughput_proxy_tokens_per_sec": 262
  },
  "_dataset": "datasets/rag.jsonl"
}
```

`quality` and `num_samples` are `null` when no quality sidecar exists. `cost.per_million_tokens` is `null` when `--cost-per-million-tokens` was not supplied to `run_eval.py`.

**Hard errors.** The advisor stops immediately (with a clear message) on:

- Tag not found in any latency directory
- `latency_tag` in quality sidecar doesn't match the loaded tag - stale sidecar can silently corrupt a recommendation
- `meta.model` in quality sidecar differs from latency result model
- Profiles being compared carry quality scores from different eval datasets - results are not comparable

**Relation to `playbook/advisor.py`.** The playbook advisor answers "what vLLM flags should I use?" The deployment advisor answers "which quantization / precision / configuration should I deploy?" Two advisors, two levels of the stack - neither replaces the other.

### 7.6 `data/generate_synthetic.py` - reference dataset

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
| `mns=8`, c=50 | 24,554ms | - | 206 tok/s |
| `mns=32`, c=50 | 7,787ms | - | 502 tok/s |
| `mns=128`, c=50 | 143ms | - | 714 tok/s |

When real data overrides a synthetic row, the report quietly upgrades. This
gives newcomers a complete-looking dataset on day 1 and a path to displace
synthetics with real measurements over time.

### 7.7 `workloads/*.yaml` - workload profiles

Three YAML workload profiles you can pass directly to `advisor.py` (or use as
config inputs in your own tooling):

| File | ISL | OSL | Concurrency | SLA | Scale |
| --- | ---: | ---: | ---: | ---: | --- |
| [`chat.yaml`](workloads/chat.yaml) | 512 | 128 | 20 | 300ms | realtime |
| [`rag.yaml`](workloads/rag.yaml) | 2048 | 256 | 30 | 700ms | mixed |
| [`long_context.yaml`](workloads/long_context.yaml) | 4096 | 512 | 50 | 2000ms | batch |

Each file also records the baseline L4/FP8 measurement for that profile so
you can sanity-check your own numbers against the validation run.

### 7.8 `examples/` - platform-specific guides

The benchmark harness is generic; deployment platforms aren't. These guides
get you from "I have an account" to "I have a benchmark JSON" on each
platform:

- **[`examples/local-vllm/QUICKSTART.md`](examples/local-vllm/QUICKSTART.md)** -
  install vLLM, serve a model, benchmark it. Covers Docker, SGLang as a drop-in
  alternative, and GPU memory requirements by model size.
- **[`examples/baseten/QUICKSTART.md`](examples/baseten/QUICKSTART.md)** -
  benchmark a Baseten-deployed model. Covers endpoint URL shape, API keys,
  cold-start warming, and how to A/B latency-optimized vs throughput-optimized
  deployments.
- **[`examples/rhoai/RUNBOOK.md`](examples/rhoai/RUNBOOK.md)** - the full
  runbook used to collect every "real" data point in this repo. Includes
  `ServingRuntime` and `InferenceService` manifests, route exposure,
  parameter-patching commands, and troubleshooting for the issues encountered
  (orphaned `LLMInferenceService` CRDs, route port-forwarding gotchas, token
  expiry).
  - [`examples/rhoai/serving_runtime.yaml`](examples/rhoai/serving_runtime.yaml) - the RHAIIS `ServingRuntime`
  - [`examples/rhoai/isvc.yaml`](examples/rhoai/isvc.yaml) - the `InferenceService` deploying Llama 3.1 8B FP8 from an OCI registry

### 7.9 `results/` - collected and reference data

Two siblings:

- `results/real/` - populated by `run_bench.py`. Gitignored by default (see
  [section 10](#10-data-lifecycle-and-gitignore)). Each file represents one
  benchmark run.
- `results/synthetic/` - populated by `generate_synthetic.py`. Committed to
  the repo so a fresh clone has working data immediately.

`report.py` reads both directories. When the same `tag` appears in both, the
real measurement wins.

---

## 8. Key findings from the validation run

Validated on **NVIDIA L4 + Llama 3.1 8B FP8** via Red Hat OpenShift AI. See
[BENCHMARK_FINDINGS.md](BENCHMARK_FINDINGS.md) for the full study including
infrastructure setup, experiment design, and per-run numbers.

| Parameter | Finding |
| --- | --- |
| `max-num-seqs` 8 → 128 | **172x** TTFT improvement at c=50 (24,554ms → 143ms) |
| Chunked prefill on L4 FP8 | No measurable benefit - FP8 eliminates the problem it solves |
| Concurrency 10 → 50 | 2.4x throughput (262 → 641 tok/s) |
| External vs internal benchmarking | ~15–30ms network floor masks sub-30ms GPU optimizations |

> **These findings are hardware-specific (NVIDIA L4, FP8).** Results will
> differ on A100/H100 with FP16 - chunked prefill in particular tends to help
> there. The tool is platform-agnostic; the *interpretation* is hardware-
> aware. `advisor.py`'s rules encode this distinction explicitly.

---

## 9. Project structure

```text
llm-inference-bench/
├── README.md                       # this file
├── BENCHMARK_FINDINGS.md           # full L4/FP8 validation study
├── requirements.txt                # aiohttp, fastapi, sqlalchemy, deepeval, pydantic, uvicorn
├── .gitignore
│
├── catalog/                        # GPU, model, and pricing specs (YAML, read at runtime)
│   ├── gpus.yaml                   # FLOPS, bandwidth, VRAM, MFU for h100/h200/a100/l40s/l4
│   ├── models.yaml                 # params, hidden dim, layers, KV heads
│   ├── pricing.yaml                # on-demand + 1-yr reserved cost per GPU-hour
│   └── anchors.yaml                # measured throughput anchors; updated by ingest_anchor.py
│
├── planner/                        # capacity planner — pure Python, no GPU needed
│   ├── capacity.py                 # roofline model: prefill + decode → replicas, TTFT, KV budget
│   ├── cost.py                     # cost envelope from catalog/pricing.yaml
│   ├── benchmark_plan.py           # ordered test matrix generator
│   ├── confidence.py               # HIGH/MEDIUM/LOW confidence rubric (±10%/25%/50%)
│   ├── ingest_anchor.py            # ingest benchmark result → calibration anchor
│   ├── compare.py                  # multi-config comparison (cheapest/safest/best-latency)
│   └── report.py                   # Markdown report with mode badge
│
├── api/                            # FastAPI REST layer with SQLite persistence
│   ├── __init__.py
│   ├── db.py                       # SQLAlchemy 2.0 ORM models (Scenario, Estimate, Run, …)
│   ├── schemas.py                  # Pydantic v2 request/response schemas
│   ├── jobs.py                     # background benchmark subprocess runner
│   └── main.py                     # create_app() factory + 9 endpoints
│
├── ui/                             # Next.js 14 App Router + Tailwind + Recharts
│   ├── package.json
│   ├── next.config.mjs             # rewrites /api/* → http://localhost:8000
│   ├── app/
│   │   ├── page.tsx                # Screen 1: Scenario Builder
│   │   ├── estimate/page.tsx       # Screen 2: Replica range chart + confidence
│   │   ├── benchmark-plan/page.tsx # Screen 3: Prioritized test matrix
│   │   └── report/page.tsx         # Screen 4: Recommendation + Markdown export
│   ├── lib/
│   │   ├── api.ts                  # API client functions
│   │   └── types.ts                # TypeScript interfaces + catalog constants
│   └── components/
│       ├── ConfidenceBadge.tsx
│       ├── ModeBadge.tsx
│       ├── ReplicaRangeChart.tsx
│       └── CopyButton.tsx
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
├── tests/                          # 268 tests total
│   ├── test_run_eval.py            # 17 tests: quality evaluator
│   ├── test_deployment_advisor.py  # 27 tests: deployment advisor
│   ├── test_capacity.py            # 28 tests: roofline model
│   ├── test_cost.py                # 18 tests: cost estimation
│   ├── test_benchmark_plan.py      # 21 tests: test matrix generation
│   ├── test_confidence.py          # 22 tests: confidence rubric + ingest
│   ├── test_api.py                 # 34 tests: FastAPI acceptance tests
│   ├── test_compare.py             # 26 tests: multi-config comparison
│   └── test_report.py              # 17 tests: Markdown report generator
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

## 10. Data lifecycle and gitignore

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

## 11. Contributing

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
   recommendation made without quality data ranks by latency only - acceptable
   for a quick filter, not for a ship decision.

Python 3.9+ compatible. `collect/run_bench.py` uses `aiohttp`; everything
else is stdlib-only.

---

## 12. License

Apache 2.0.
