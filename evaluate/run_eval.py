#!/usr/bin/env python3
"""
Offline quality evaluator for LLM inference deployments.

Sends a small evaluation dataset at an OpenAI-compatible endpoint,
scores responses with DeepEval, and writes a quality sidecar JSON
alongside the latency result.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


def load_dataset(path):
    """Load and validate JSONL dataset. Returns list of row dicts."""
    required = {"schema_version", "id", "workload", "prompt", "expected"}
    valid_workloads = {"chat", "rag", "long_context"}
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print("ERROR: {}:{}: invalid JSON: {}".format(path, lineno, e), file=sys.stderr)
                sys.exit(1)
            missing = required - set(row.keys())
            if missing:
                print("ERROR: {}:{}: missing fields: {}".format(path, lineno, missing), file=sys.stderr)
                sys.exit(1)
            if row["workload"] not in valid_workloads:
                print("ERROR: {}:{}: unknown workload '{}'".format(path, lineno, row["workload"]), file=sys.stderr)
                sys.exit(1)
            rows.append(row)
    if not rows:
        print("ERROR: {}: no valid rows found".format(path), file=sys.stderr)
        sys.exit(1)
    return rows


def normalize_score(metric_name, raw_score):
    """Normalize metric to higher-is-better in range [0, 1].
    Inverts rate metrics where lower is better (e.g. hallucination_rate)."""
    if metric_name == "hallucination":
        return max(0.0, 1.0 - float(raw_score))
    return float(raw_score)


def select_metrics(workload, has_contexts):
    """Return list of metric names to activate for this workload."""
    metrics = ["answer_relevancy", "correctness"]
    if workload == "rag" and has_contexts:
        metrics += ["faithfulness", "hallucination"]
    return metrics


def derive_tag(latency_result_path):
    """Derive output tag from latency result filename."""
    return os.path.basename(latency_result_path).replace(".json", "")


def write_sidecar(out_dir, tag, latency_tag, evaluator, model, dataset_path,
                  num_samples, metrics, overall_score, cost_per_million, throughput_proxy):
    """Write quality sidecar JSON to <out_dir>/<tag>.json."""
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "meta": {
            "tag": tag,
            "latency_tag": latency_tag,
            "evaluator": evaluator,
            "model": model,
            "dataset": dataset_path,
            "num_samples": num_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": dict(metrics, overall_score=overall_score),
        "cost": {
            "per_million_tokens": cost_per_million,
            "throughput_proxy_tokens_per_sec": throughput_proxy,
        },
    }
    path = os.path.join(out_dir, tag + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


async def send_prompt(session, endpoint, model, token, prompt, max_tokens=256):
    """Send a single prompt (non-streaming) and return the response text."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    async with session.post(endpoint, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError("HTTP {}".format(resp.status))
        data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("empty choices in response")
        return (
            choices[0].get("text")
            or choices[0].get("message", {}).get("content")
            or ""
        )


async def collect_responses(endpoint, model, token, dataset, concurrency=5):
    """
    Send all dataset prompts to the endpoint.
    Returns (samples, errors) where samples is a list of (row, response_text) tuples.
    Concurrency is deliberately low (5) to avoid warming the KV cache or
    interfering with a parallel load test.
    """
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=120)
    samples = []
    errors = []

    async def one(session, row):
        async with sem:
            try:
                response = await send_prompt(session, endpoint, model, token, row["prompt"])
                samples.append((row, response))
            except Exception as e:
                errors.append((row["id"], repr(e)))
                print("WARN: sample {} failed: {}".format(row["id"], e), file=sys.stderr)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*[one(session, row) for row in dataset])

    return samples, errors


def run_deepeval(samples, eval_model, workload):
    """
    Score (row, response) samples using DeepEval metrics.
    Returns (aggregated_metrics_dict, overall_score).
    All metrics normalized to higher-is-better before averaging.
    """
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    has_contexts = any(row.get("contexts") for row, _ in samples)
    active = select_metrics(workload, has_contexts)

    metrics = []
    if "answer_relevancy" in active:
        metrics.append(AnswerRelevancyMetric(model=eval_model, threshold=0.5))
    if "correctness" in active:
        metrics.append(GEval(
            name="Correctness",
            criteria=(
                "Does the actual output accurately answer the input question "
                "based on the expected output?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=eval_model,
            threshold=0.5,
        ))
    if "faithfulness" in active:
        metrics.append(FaithfulnessMetric(model=eval_model, threshold=0.5))
    if "hallucination" in active:
        metrics.append(HallucinationMetric(model=eval_model, threshold=0.5))

    def canonical_name(metric):
        n = type(metric).__name__.lower()
        if "relevancy" in n:
            return "answer_relevancy"
        if "geval" in n or "correctness" in n:
            return "correctness"
        if "faithfulness" in n:
            return "faithfulness"
        if "hallucination" in n:
            return "hallucination"
        return n

    per_metric = {m: [] for m in active}

    for row, response in samples:
        tc = LLMTestCase(
            input=row["prompt"],
            actual_output=response,
            expected_output=row["expected"],
            context=row.get("contexts"),
        )
        for metric in metrics:
            try:
                metric.measure(tc)
                key = canonical_name(metric)
                if key in per_metric:
                    per_metric[key].append(normalize_score(key, metric.score))
            except Exception as e:
                print(
                    "WARN: DeepEval {} failed on sample {}: {}".format(
                        type(metric).__name__, row["id"], e
                    ),
                    file=sys.stderr,
                )

    aggregated = {
        k: round(sum(vals) / len(vals), 4)
        for k, vals in per_metric.items()
        if vals
    }
    overall_score = (
        round(sum(aggregated.values()) / len(aggregated), 4) if aggregated else None
    )
    return aggregated, overall_score


def run_llm_judge(samples, eval_endpoint, eval_model, eval_token):
    """
    Score samples using any OpenAI-compatible chat endpoint as judge.
    Returns (aggregated_metrics_dict, overall_score).
    """
    import urllib.request

    SCORING_PROMPT = (
        "Score this response on three dimensions (1-5 each):\n"
        "  correctness: does it answer the question accurately?\n"
        "  helpfulness: is it useful and complete?\n"
        "  hallucination: 5=none, 1=severe fabrication\n\n"
        "Question: {prompt}\n"
        "Expected answer: {expected}\n"
        "Response: {response}\n\n"
        'Return JSON only: {{"correctness": N, "helpfulness": N, "hallucination": N}}'
    )

    scores = {"correctness": [], "helpfulness": [], "hallucination": []}

    for row, response in samples:
        prompt = SCORING_PROMPT.format(
            prompt=row["prompt"][:500],
            expected=row["expected"][:500],
            response=response[:500],
        )
        headers = {"Content-Type": "application/json"}
        if eval_token:
            headers["Authorization"] = "Bearer " + eval_token
        payload = json.dumps({
            "model": eval_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 64,
        }).encode()

        try:
            req = urllib.request.Request(
                eval_endpoint.rstrip("/") + "/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            for k in scores:
                if k in parsed:
                    raw = float(parsed[k]) / 5.0  # normalize 1-5 → 0-1
                    scores[k].append(normalize_score(k, raw))
        except Exception as e:
            print("WARN: LLM judge failed on sample {}: {}".format(row["id"], e), file=sys.stderr)

    aggregated = {
        k: round(sum(vals) / len(vals), 4)
        for k, vals in scores.items()
        if vals
    }
    overall_score = (
        round(sum(aggregated.values()) / len(aggregated), 4) if aggregated else None
    )
    return aggregated, overall_score


def main():
    ap = argparse.ArgumentParser(description="Offline quality evaluator for LLM deployments")
    ap.add_argument("--endpoint", required=True,
                    help="Inference endpoint being evaluated (OpenAI-compatible /v1/completions)")
    ap.add_argument("--model", required=True,
                    help="Model name served at --endpoint")
    ap.add_argument("--latency-result", required=True, dest="latency_result",
                    help="Path to the latency JSON produced by collect/run_bench.py")
    ap.add_argument("--dataset", required=True,
                    help="Path to eval JSONL dataset (datasets/chat.jsonl etc.)")
    ap.add_argument("--evaluator", choices=["deepeval", "llm-judge"], default="deepeval",
                    help="Scoring backend (default: deepeval)")
    ap.add_argument("--eval-model", dest="eval_model", default="gpt-4o",
                    help="Judge model name used by DeepEval or llm-judge (default: gpt-4o)")
    ap.add_argument("--eval-endpoint", dest="eval_endpoint",
                    default="https://api.openai.com/v1",
                    help="Judge model endpoint (default: https://api.openai.com/v1)")
    ap.add_argument("--token", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="Bearer token for --endpoint (default: $OPENAI_API_KEY)")
    ap.add_argument("--eval-token", dest="eval_token",
                    default=os.environ.get("OPENAI_API_KEY", ""),
                    help="Bearer token for --eval-endpoint (default: $OPENAI_API_KEY)")
    ap.add_argument("--cost-per-million-tokens", type=float, dest="cost_per_million",
                    default=None,
                    help="Cost per 1M output tokens for this deployment (optional)")
    ap.add_argument("--output-dir", dest="output_dir", default="./results/quality",
                    help="Directory for quality sidecar JSON (default: ./results/quality)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Validate inputs and print plan without hitting the endpoint")
    args = ap.parse_args()

    if not os.path.isfile(args.latency_result):
        print("ERROR: latency result not found: {}".format(args.latency_result), file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.dataset):
        print("ERROR: dataset not found: {}".format(args.dataset), file=sys.stderr)
        sys.exit(1)

    with open(args.latency_result) as f:
        latency_data = json.load(f)
    latency_tag = latency_data.get("meta", {}).get("tag") or derive_tag(args.latency_result)
    throughput_proxy = latency_data.get("metrics", {}).get("throughput_tokens_per_sec")

    dataset = load_dataset(args.dataset)
    tag = derive_tag(args.latency_result)
    workloads = {row["workload"] for row in dataset}
    workload = workloads.pop() if len(workloads) == 1 else "mixed"

    if args.dry_run:
        print("=== Dry run ===")
        print("  latency-result : {}".format(args.latency_result))
        print("  latency-tag    : {}".format(latency_tag))
        print("  dataset        : {} ({} samples, workload={})".format(
            args.dataset, len(dataset), workload))
        print("  evaluator      : {}".format(args.evaluator))
        print("  eval-model     : {}".format(args.eval_model))
        print("  eval-endpoint  : {}".format(args.eval_endpoint))
        print("  output-tag     : {}".format(tag))
        print("  output-dir     : {}".format(args.output_dir))
        print("Would collect responses and run evaluation. Exiting (--dry-run).")
        return

    print("Collecting {} responses from {} ...".format(len(dataset), args.endpoint))
    samples, errors = asyncio.run(
        collect_responses(args.endpoint, args.model, args.token, dataset)
    )
    if not samples:
        print("ERROR: no samples collected — check endpoint and model", file=sys.stderr)
        sys.exit(1)
    if errors:
        print("WARN: {}/{} samples failed, continuing with {}".format(
            len(errors), len(dataset), len(samples)))

    print("Evaluating with {} ...".format(args.evaluator))
    if args.evaluator == "deepeval":
        try:
            import deepeval  # noqa: F401
        except ImportError:
            print("ERROR: DeepEval not installed. Run: pip install deepeval", file=sys.stderr)
            sys.exit(1)
        metrics, overall_score = run_deepeval(samples, args.eval_model, workload)
    else:
        metrics, overall_score = run_llm_judge(
            samples, args.eval_endpoint, args.eval_model, args.eval_token
        )

    out_path = write_sidecar(
        out_dir=os.path.expanduser(args.output_dir),
        tag=tag,
        latency_tag=latency_tag,
        evaluator=args.evaluator,
        model=args.model,
        dataset_path=args.dataset,
        num_samples=len(samples),
        metrics=metrics,
        overall_score=overall_score,
        cost_per_million=args.cost_per_million,
        throughput_proxy=throughput_proxy,
    )

    print("overall_score={}".format(overall_score))
    for k, v in metrics.items():
        print("  {}={}".format(k, v))
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
