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
