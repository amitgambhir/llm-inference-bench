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
