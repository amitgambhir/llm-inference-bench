#!/usr/bin/env python3
"""
Generate synthetic benchmark results grounded in real L4/FP8 measurements.

All output JSON is tagged synthetic=true. Real anchors come from the
validated NVIDIA L4 FP8 run documented in BENCHMARK_FINDINGS.md. Other
scenarios (A100/H100 FP16, SGLang, ISL/concurrency sweeps) are extrapolated
from those anchors using simple, conservative models so the reference data
stays consistent with reality.

Stdlib only — no aiohttp required.
"""
import json
import os
from datetime import datetime, timezone

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

TS = datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(meta, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, meta["tag"] + ".json")
    obj = {"meta": meta, "metrics": metrics}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def base_meta(tag, runtime="vllm", model="llama-3.1-8b", gpu_name="NVIDIA L4",
              mem_mb=23034, isl=2048, osl=128, c=10, dur=90,
              chunked_prefill=False, shared_prefix=False):
    return {
        "tag": tag,
        "runtime": runtime,
        "model": model,
        "gpu": {"name": gpu_name, "memory_mb": mem_mb, "util_pct": 0},
        "config": {
            "chunked_prefill": chunked_prefill,
            "tensor_parallel_size": 1,
            "shared_prefix": shared_prefix,
        },
        "workload": {
            "isl_approx": isl,
            "osl_max": osl,
            "concurrency": c,
            "duration_secs": dur,
        },
        "synthetic": True,
        "timestamp": TS,
    }


def metrics(ttft_p50, ttft_p95, throughput_tok, c=10, dur=90, p99_mul=1.5,
            mean_factor=0.96, total_lat_p50=None, total_lat_p95=None,
            failed=0):
    if total_lat_p50 is None:
        total_lat_p50 = ttft_p50 + 4500
    if total_lat_p95 is None:
        total_lat_p95 = ttft_p95 + 4600
    req_per_sec = throughput_tok / 128.0
    total = max(1, int(round(req_per_sec * dur)))
    return {
        "ttft_ms": {
            "p50": round(ttft_p50, 1),
            "p90": round((ttft_p50 + ttft_p95) / 2.0, 1),
            "p95": round(ttft_p95, 1),
            "p99": round(ttft_p95 * p99_mul, 1),
            "mean": round(ttft_p50 * mean_factor, 1),
        },
        "total_latency_ms": {
            "p50": round(total_lat_p50, 1),
            "p95": round(total_lat_p95, 1),
            "p99": round(total_lat_p95 * 1.05, 1),
        },
        "throughput_tokens_per_sec": round(throughput_tok, 1),
        "throughput_req_per_sec": round(req_per_sec, 2),
        "total_requests": total + failed,
        "successful_requests": total,
        "failed_requests": failed,
    }


def l4_fp8_baseline():
    # ISL=512, c=10 — TTFT p50=75, p95=81, throughput=262
    write(base_meta("vllm_l4fp8_isl512_c10", isl=512),
          metrics(75, 81, 262))
    # ISL=2048, c=10 — TTFT p50=115, p95=133, throughput=262
    write(base_meta("vllm_l4fp8_isl2k_c10", isl=2048),
          metrics(115, 133, 262))
    # ISL=4096, c=50 — TTFT p50=134, p95=335, throughput=641
    write(base_meta("vllm_l4fp8_isl4k_c50", isl=4096, c=50),
          metrics(134, 335, 641, c=50))


def max_num_seqs_sweep():
    # All at c=50, ISL=2048, illustrating mns headline finding
    # mns=8: TTFT p50=24554, throughput=206
    m = base_meta("vllm_l4fp8_isl2k_c50_mns8", c=50)
    m["config"]["max_num_seqs"] = 8
    write(m, metrics(24554, 26100, 206, c=50,
                     total_lat_p50=27000, total_lat_p95=29000))
    # mns=32: TTFT p50=7787, throughput=502
    m = base_meta("vllm_l4fp8_isl2k_c50_mns32", c=50)
    m["config"]["max_num_seqs"] = 32
    write(m, metrics(7787, 9100, 502, c=50,
                     total_lat_p50=11000, total_lat_p95=12500))
    # mns=128: TTFT p50=143, throughput=714
    m = base_meta("vllm_l4fp8_isl2k_c50_mns128", c=50)
    m["config"]["max_num_seqs"] = 128
    write(m, metrics(143, 380, 714, c=50,
                     total_lat_p50=4700, total_lat_p95=4900))


def chunked_prefill_l4_fp8():
    # L4 FP8 chunked-prefill: no measurable benefit
    m_off = base_meta("vllm_l4fp8_isl4k_c50_cp_off", isl=4096, c=50,
                      chunked_prefill=False)
    write(m_off, metrics(134, 335, 641, c=50))
    m_on = base_meta("vllm_l4fp8_isl4k_c50_cp_on", isl=4096, c=50,
                     chunked_prefill=True)
    write(m_on, metrics(136, 342, 638, c=50))


def chunked_prefill_a100_fp16():
    # Literature: 3-5x p95 improvement at high ISL on A100/H100 FP16
    m_off = base_meta("vllm_a100fp16_isl4k_c50_cp_off", isl=4096, c=50,
                      gpu_name="NVIDIA A100", mem_mb=40960,
                      model="llama-3.1-8b-fp16")
    write(m_off, metrics(310, 1450, 980, c=50,
                         total_lat_p50=4900, total_lat_p95=6100))
    m_on = base_meta("vllm_a100fp16_isl4k_c50_cp_on", isl=4096, c=50,
                     gpu_name="NVIDIA A100", mem_mb=40960,
                     model="llama-3.1-8b-fp16", chunked_prefill=True)
    write(m_on, metrics(290, 410, 1180, c=50,
                        total_lat_p50=4700, total_lat_p95=4900))


def h100_fp16():
    write(base_meta("vllm_h100fp16_isl2k_c10", gpu_name="NVIDIA H100",
                    mem_mb=81920, model="llama-3.1-8b-fp16"),
          metrics(58, 72, 540))
    write(base_meta("vllm_h100fp16_isl2k_c50", c=50, gpu_name="NVIDIA H100",
                    mem_mb=81920, model="llama-3.1-8b-fp16"),
          metrics(95, 210, 1480, c=50))


def sglang_comparison():
    # ~15% higher throughput than vLLM on equivalent hardware
    write(base_meta("sglang_l4fp8_isl2k_c10", runtime="sglang"),
          metrics(112, 130, 301))
    write(base_meta("sglang_l4fp8_isl4k_c50", runtime="sglang",
                    isl=4096, c=50),
          metrics(128, 320, 737, c=50))


def concurrency_sweep():
    # ISL=2048, mns=128
    for c, ttft50, ttft95, tok in [
        (1, 95, 110, 38),
        (5, 105, 122, 175),
        (10, 115, 133, 262),
        (20, 120, 155, 410),
        (50, 143, 380, 641),
    ]:
        m = base_meta("vllm_l4fp8_isl2k_c{}".format(c), c=c)
        m["config"]["max_num_seqs"] = 128
        write(m, metrics(ttft50, ttft95, tok, c=c))


def isl_interpolation():
    # mns=128, c=20, varying ISL
    for isl, ttft50, ttft95, tok in [
        (1024, 95, 110, 380),
        (3072, 125, 175, 380),
        (8192, 220, 480, 320),
    ]:
        m = base_meta("vllm_l4fp8_isl{}_c20".format(isl), isl=isl, c=20)
        m["config"]["max_num_seqs"] = 128
        write(m, metrics(ttft50, ttft95, tok, c=20))


def prefix_caching():
    # External-benchmark view: marginal improvement (network floor masks GPU savings)
    write(base_meta("vllm_l4fp8_isl2k_c10_prefix_off", isl=2048, shared_prefix=False),
          metrics(115, 133, 262))
    write(base_meta("vllm_l4fp8_isl2k_c10_prefix_on", isl=2048, shared_prefix=True),
          metrics(108, 128, 270))


def main():
    l4_fp8_baseline()
    max_num_seqs_sweep()
    chunked_prefill_l4_fp8()
    chunked_prefill_a100_fp16()
    h100_fp16()
    sglang_comparison()
    concurrency_sweep()
    isl_interpolation()
    prefix_caching()
    files = sorted(os.listdir(OUTPUT_DIR))
    files = [f for f in files if f.endswith(".json")]
    print("wrote {} synthetic result files to {}".format(len(files), OUTPUT_DIR))


if __name__ == "__main__":
    main()
