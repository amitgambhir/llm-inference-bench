#!/usr/bin/env python3
"""
Hardware-aware vLLM configuration advisor.

Translates a workload description (ISL, latency SLA, concurrency, scale) plus
target hardware into a concrete configuration recommendation. The rules are
grounded in real measurements from the L4/FP8 validation run documented in
BENCHMARK_FINDINGS.md, and extrapolated conservatively to other GPUs.

Stdlib only.
"""
import argparse
import json
import math
import sys


GPUS = ("l4", "l40s", "a100", "h100")
PRECISIONS = ("fp8", "fp16")
SCALES = ("realtime", "mixed", "batch")


def recommend_max_num_seqs(concurrency):
    val = max(concurrency * 2, 64)
    return min(val, 256)


def recommend_chunked_prefill(gpu, precision, isl):
    """
    Returns (enable: bool, rationale: str).
    """
    if precision == "fp8" and gpu in ("l4", "l40s"):
        return (False,
                "Real data: ISL=4096, c=50 on L4 FP8 showed zero benefit from "
                "chunked prefill. FP8 makes prefill fast enough that "
                "monopolization is not the bottleneck the flag targets.")
    if precision == "fp16" and gpu in ("a100", "h100") and isl > 1024:
        return (True,
                "Literature: 3–5x p95 improvement at high ISL on A100/H100 FP16 "
                "where prefill cost dominates and monopolizes the batch.")
    return (False,
            "Hardware/precision/ISL combination is not strongly characterized — "
            "leave disabled and measure both ways if latency matters.")


def recommend_prefix_caching():
    return (True,
            "Always enable. No measurable downside. External benchmarks may "
            "not show the win due to network floor — the GPU-level savings are "
            "real and visible internally.")


def per_replica_rps(isl):
    # L4 FP8 anchor: ~5.5 req/s sustainable at ISL=2048 with mns=128.
    # Capacity scales sub-linearly with ISL (prefill dominates at high ISL).
    base = 5.5
    isl_factor = (2048.0 / max(isl, 256)) ** 0.5
    return base * isl_factor


def replica_estimate(target_rps, isl, scale):
    sustainable = per_replica_rps(isl)
    # 40% headroom for realtime per spec, 25% for mixed, 10% for batch
    headroom = 0.6 if scale == "realtime" else (0.75 if scale == "mixed" else 0.9)
    effective = sustainable * headroom
    if effective <= 0:
        return 1
    return max(1, int(math.ceil(target_rps / effective)))


def build_recommendation(args):
    warnings = []
    notes = []

    mns = recommend_max_num_seqs(args.concurrency)
    cp_enable, cp_rationale = recommend_chunked_prefill(args.gpu, args.model_precision, args.isl)
    prefix_enable, prefix_rationale = recommend_prefix_caching()

    # Real data callout for mns
    mns_rationale = (
        "Set to max(concurrency * 2, 64), capped at 256. "
        "Real data: at c=50 on L4 FP8, mns=8 gave TTFT p50=24,554ms vs mns=128 "
        "at 143ms — a 172x improvement. This is the single most impactful parameter."
    )

    # Warnings
    if args.concurrency > mns:
        warnings.append(
            "Concurrency ({}) exceeds the recommended max-num-seqs ({}). "
            "Requests will queue and TTFT will spike. Raise mns or reduce "
            "concurrency.".format(args.concurrency, mns))
    if args.gpu == "l4" and args.model_precision == "fp16":
        warnings.append(
            "L4 has 23GB VRAM. FP16 8B+ models leave little headroom for KV cache; "
            "expect OOMs at high ISL/concurrency. FP8 is strongly preferred on L4.")
    if args.scale == "realtime" and args.isl > 4096:
        warnings.append(
            "Realtime SLA with ISL>4096 is aggressive — prefill alone may "
            "exceed your latency budget. Validate with a real measurement before committing.")

    target_rps = args.concurrency / max(args.latency_sla / 1000.0, 0.05)
    replicas = replica_estimate(target_rps, args.isl, args.scale)

    return {
        "inputs": {
            "isl": args.isl,
            "latency_sla_ms": args.latency_sla,
            "concurrency": args.concurrency,
            "scale": args.scale,
            "gpu": args.gpu,
            "model_precision": args.model_precision,
        },
        "config": {
            "runtime": "vllm",
            "max_num_seqs": mns,
            "enable_chunked_prefill": cp_enable,
            "enable_prefix_caching": prefix_enable,
            "tensor_parallel_size": 1,
        },
        "rationale": {
            "max_num_seqs": mns_rationale,
            "chunked_prefill": cp_rationale,
            "prefix_caching": prefix_rationale,
        },
        "capacity": {
            "target_rps": round(target_rps, 2),
            "per_replica_rps": round(per_replica_rps(args.isl), 2),
            "replicas": replicas,
            "headroom_policy": (
                "60% for realtime, 75% for mixed, 90% for batch"),
        },
        "warnings": warnings,
        "notes": notes,
    }


def print_human(rec):
    print("=" * 60)
    print("vLLM Configuration Recommendation")
    print("=" * 60)
    i = rec["inputs"]
    print("Workload: ISL={}  SLA={}ms  Concurrency={}  Scale={}".format(
        i["isl"], i["latency_sla_ms"], i["concurrency"], i["scale"]))
    print("Hardware: {} ({})".format(i["gpu"].upper(), i["model_precision"].upper()))
    print()

    print("Recommended config:")
    c = rec["config"]
    print("  runtime: {}".format(c["runtime"]))
    print("  --max-num-seqs={}".format(c["max_num_seqs"]))
    flag = "(enabled)" if c["enable_chunked_prefill"] else "(disabled)"
    print("  --enable-chunked-prefill  {}".format(flag))
    print("  --enable-prefix-caching   (enabled)")
    print("  --tensor-parallel-size={}".format(c["tensor_parallel_size"]))
    print()

    print("Rationale:")
    for k, v in rec["rationale"].items():
        print("  [{}]".format(k))
        print("    " + v.replace("\n", "\n    "))
    print()

    cap = rec["capacity"]
    print("Capacity estimate:")
    print("  target RPS:        {}".format(cap["target_rps"]))
    print("  per-replica RPS:   {} (sustainable)".format(cap["per_replica_rps"]))
    print("  replicas needed:   {}".format(cap["replicas"]))
    print("  headroom:          {}".format(cap["headroom_policy"]))
    print()

    if rec["warnings"]:
        print("WARNINGS:")
        for w in rec["warnings"]:
            print("  ! " + w)
        print()


def interactive_inputs():
    def ask(prompt, default, cast=str, choices=None):
        while True:
            s = input("{} [{}]: ".format(prompt, default)).strip()
            if not s:
                s = str(default)
            if choices and s not in choices:
                print("  choose one of: " + ", ".join(choices))
                continue
            try:
                return cast(s)
            except ValueError:
                print("  invalid value")
    print("Interactive mode — press Enter to accept defaults.")
    return argparse.Namespace(
        isl=ask("ISL (tokens)", 2048, int),
        latency_sla=ask("Latency SLA p95 (ms)", 700, int),
        concurrency=ask("Concurrency", 20, int),
        scale=ask("Scale", "mixed", str, SCALES),
        gpu=ask("GPU", "l4", str, GPUS),
        model_precision=ask("Model precision", "fp8", str, PRECISIONS),
        interactive=True,
        json=False,
    )


def main():
    ap = argparse.ArgumentParser(description="vLLM config advisor")
    ap.add_argument("--isl", type=int)
    ap.add_argument("--latency-sla", type=int, dest="latency_sla")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--scale", choices=SCALES)
    ap.add_argument("--gpu", choices=GPUS, default="l4")
    ap.add_argument("--model-precision", choices=PRECISIONS, default="fp8",
                    dest="model_precision")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        args = interactive_inputs()
    else:
        missing = [n for n in ("isl", "latency_sla", "concurrency", "scale")
                   if getattr(args, n) is None]
        if missing:
            print("Missing required args: --{}".format(", --".join(missing).replace("_", "-")),
                  file=sys.stderr)
            ap.print_help(sys.stderr)
            sys.exit(2)

    rec = build_recommendation(args)
    if args.json:
        json.dump(rec, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_human(rec)


if __name__ == "__main__":
    main()
