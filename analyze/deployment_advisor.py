#!/usr/bin/env python3
"""
Quality-aware deployment advisor.

Merges latency benchmark results with quality evaluation sidecars to
produce a deployment recommendation balancing latency, cost, and quality.
"""
import json
import os
import sys


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DIR = os.path.join(REPO_ROOT, "results", "real")
SYN_DIR = os.path.join(REPO_ROOT, "results", "synthetic")
QUALITY_DIR = os.path.join(REPO_ROOT, "results", "quality")


def _find_latency_file(tag, latency_dirs):
    """Return path to latency JSON for tag. Later dirs override earlier ones."""
    found = None
    for d in latency_dirs:
        path = os.path.join(d, tag + ".json")
        if os.path.isfile(path):
            found = path
    return found


def validate_profile(profile):
    """Fail fast if required latency fields are missing or None."""
    required = {"ttft_ms_p50", "ttft_ms_p95", "throughput_tokens_per_sec"}
    lat = profile.get("latency", {})
    missing = {f for f in required if lat.get(f) is None}
    if missing:
        print(
            "ERROR: profile '{}' missing latency fields: {}".format(
                profile.get("tag"), missing
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def load_deployment(tag, latency_dirs, quality_dir):
    """
    Load and merge latency + quality data into a normalized DeploymentProfile.

    Flattens the existing nested result schema (e.g. metrics.ttft_ms.p50)
    into a flat in-memory shape (latency.ttft_ms_p50) so all downstream
    functions work against a single consistent structure.
    """
    lat_path = _find_latency_file(tag, latency_dirs)
    if lat_path is None:
        print("ERROR: no latency result found for tag '{}'".format(tag), file=sys.stderr)
        sys.exit(1)

    with open(lat_path) as f:
        lat_raw = json.load(f)

    meta = lat_raw.get("meta", {})
    m = lat_raw.get("metrics", {})
    ttft = m.get("ttft_ms", {})

    profile = {
        "tag": tag,
        "model": meta.get("model", "unknown"),
        "latency": {
            "ttft_ms_p50": ttft.get("p50"),
            "ttft_ms_p95": ttft.get("p95"),
            "throughput_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "quality": None,
        "cost": {
            "per_million_tokens": None,
            "throughput_proxy_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "_dataset": None,
    }
    validate_profile(profile)

    qual_path = os.path.join(quality_dir, tag + ".json")
    if os.path.isfile(qual_path):
        with open(qual_path) as f:
            qual_raw = json.load(f)

        qual_latency_tag = qual_raw.get("meta", {}).get("latency_tag")
        if qual_latency_tag and qual_latency_tag != tag:
            print(
                "ERROR: quality sidecar for '{}' has latency_tag='{}'. "
                "This sidecar was generated for a different latency result. "
                "Re-run evaluate/run_eval.py with --latency-result pointing "
                "to the correct file.".format(tag, qual_latency_tag),
                file=sys.stderr,
            )
            sys.exit(1)

        qm = qual_raw.get("metrics", {})
        profile["quality"] = {
            "overall_score": qm.get("overall_score"),
            "metrics": {k: v for k, v in qm.items() if k != "overall_score"},
        }
        cost = qual_raw.get("cost", {})
        profile["cost"]["per_million_tokens"] = cost.get("per_million_tokens")

        # prefer quality sidecar value; fall back to latency file throughput
        profile["cost"]["throughput_proxy_tokens_per_sec"] = (
            cost.get("throughput_proxy_tokens_per_sec")
            or profile["cost"]["throughput_proxy_tokens_per_sec"]
        )
        profile["_dataset"] = qual_raw.get("meta", {}).get("dataset")
    else:
        print("WARN: no quality sidecar for '{}' (expected {}) — quality metrics will be N/A".format(tag, qual_path), file=sys.stderr)

    return profile
