#!/usr/bin/env python3
"""
Platform-agnostic async LLM benchmark harness.

Targets any OpenAI-compatible /v1/completions endpoint (local vLLM, SGLang,
Baseten, RHOAI, SageMaker, Vertex, Azure ML, etc.). Streams responses to
measure TTFT precisely.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


ENDPOINT = ""
MODEL = ""
TOKEN = ""
BASIC_AUTH: "aiohttp.BasicAuth | None" = None
USE_SHARED_PREFIX = False


SHARED_PREFIX = (
    "You are a helpful enterprise assistant. Follow company policy. "
    "Be concise, accurate, and reference any provided documents. "
)


PROMPT_512 = (
    "Customer support ticket #48211. Customer: Jane Doe, policy holder since 2019. "
    "Issue: Her smart thermostat (model NX-22, purchased March 2024) is showing "
    "error code E7 and is not connecting to her home WiFi. She has already restarted "
    "the device twice and reset the WiFi router. She is enrolled in our Home Tech "
    "Protection plan which covers diagnostics and replacement. Previous tickets: "
    "two minor billing questions in 2023, both resolved within 24 hours. "
    "She prefers email communication and is available between 6pm and 9pm EST. "
    "Task: draft a response that acknowledges the issue, explains the next two "
    "troubleshooting steps (factory reset and firmware check), and offers a "
    "replacement under warranty if those steps fail. Keep tone friendly and "
    "professional. Reference her plan benefits explicitly."
)


PROMPT_2048 = (
    "Enterprise claim review case file. Claim ID: CLM-2026-04-19388. "
    "Policyholder: Acme Industrial Manufacturing, commercial property policy "
    "CPP-887-2024, in force since January 2020. Annual premium $487,000. "
    "Loss event: water damage to finished goods warehouse, reported April 14 2026. "
    "Estimated loss: $1.42M (inventory) + $312K (building repairs). "
    "Cause as reported by insured: sprinkler pipe burst during overnight freeze "
    "(temperatures recorded at -8F by nearest weather station). Adjuster on site "
    "April 16: confirmed pipe failure at joint near loading bay, photographed "
    "ice formation patterns consistent with freezing. Building HVAC logs show "
    "the loading bay heater was offline from April 11 through April 15 (work "
    "order WO-2026-2841 indicates a scheduled repair that ran four days behind). "
    "Policy language: section 4.2 excludes losses 'resulting from the insured's "
    "failure to maintain heat in covered premises during freezing conditions, "
    "where such failure is not due to peril otherwise covered.' Section 4.3 "
    "however reinstates coverage where the maintenance failure was 'unforeseeable "
    "or beyond reasonable control of the insured.' Prior similar claims: none "
    "for this policyholder in the last five years. Loss control survey from "
    "September 2025 noted the HVAC contractor relationship as 'satisfactory' "
    "and identified no exclusion-triggering risks. The insured argues the four-day "
    "repair delay was caused by their contractor missing parts shipments, which "
    "they could not have anticipated. The adjuster's preliminary recommendation "
    "is partial coverage at 60% pending legal review. Underwriting wants to "
    "preserve the relationship — Acme is in renewal discussions for a multi-line "
    "package worth approximately $1.2M annually. Compliance has flagged that "
    "any settlement above $750K requires senior claims officer approval and a "
    "formal coverage opinion memorandum. Task: produce a written coverage "
    "analysis. Identify whether 4.2 exclusion or 4.3 reinstatement controls. "
    "Recommend a settlement posture (full pay, partial pay with reservation, "
    "or denial) with rationale grounded in the policy language and the facts "
    "above. Flag any additional investigation steps that would materially "
    "change the analysis. Format as a structured memo: Issue / Facts / "
    "Analysis / Recommendation."
)


PROMPT_4096 = (
    "Quarterly enterprise risk portfolio review. Reviewing analyst: senior "
    "underwriter, commercial lines. Period: Q1 2026. The following four case "
    "files must be reviewed together to identify portfolio-level patterns, "
    "concentration risks, and compliance gaps. Each case file is provided in "
    "full. After reviewing all four, produce a portfolio-level analysis.\n\n"
    "Case 1 — Manufacturing concentration: Acme Industrial (CPP-887-2024), "
    "premium $487K, water damage claim of $1.7M in April. HVAC maintenance gap "
    "implicated. Section 4.2 exclusion under review. Acme is also one of seven "
    "metals-fabrication accounts in the portfolio, total premium $3.1M. Three "
    "of those seven accounts share the same regional HVAC contractor (NorthernPlex "
    "Mechanical) which has been associated with two prior maintenance-related "
    "claims in 2024. Compliance flag: vendor concentration risk not previously "
    "documented in portfolio review. Action item from January: 'review shared-"
    "vendor exposure across metals book' — not yet completed.\n\n"
    "Case 2 — Tech sector cyber: DataMesh Cloud Services (CYB-441-2025), "
    "premium $612K, ransomware claim of $4.2M reported March 8. Threat actor "
    "encrypted production databases for 11 days. Insured paid no ransom; "
    "recovery was completed from offline backups. Business interruption portion "
    "$2.8M, forensic and notification $1.4M. Policy includes $5M aggregate "
    "with $250K SIR. Sub-limit on regulatory fines $1M. Pending: state AG "
    "investigation into notification timing (DataMesh notified affected "
    "customers on day 9, statute requires within 7 days from confirmed "
    "exfiltration, but DataMesh argues exfiltration was never confirmed — "
    "only encryption). Coverage question: does the notification timing dispute "
    "trigger the regulatory exclusion in section 7.4? Three other cyber accounts "
    "in the portfolio share the same managed-service provider as DataMesh "
    "(Aegis IT Partners), creating provider-level concentration of $1.9M "
    "aggregate premium.\n\n"
    "Case 3 — Healthcare professional liability: Riverside Specialty Group "
    "(PRO-220-2023), premium $341K, malpractice suit filed February 19 alleging "
    "delayed cancer diagnosis. Plaintiff demand $3.5M. Policy limits $5M per "
    "claim, $7M aggregate. Defense panel engaged. Discovery is in early stages. "
    "Notable: Riverside has had three plaintiff demands above $1M in the past "
    "24 months. Loss ratio on this account now 87% over rolling three years. "
    "Underwriting flag from December: 'evaluate non-renewal vs rate increase + "
    "tighter conditions.' Pending renewal date: October 2026. Healthcare book "
    "overall: $14.2M premium, loss ratio 71% trailing twelve months, above the "
    "target of 62%.\n\n"
    "Case 4 — Energy property: Cascade Pipeline Operations (PRP-918-2024), "
    "premium $1.1M, no open claims this quarter but a near-miss incident "
    "reported voluntarily on February 27. Routine inspection identified a "
    "corrosion-related thinning on a 14-inch transmission line, repaired before "
    "any release occurred. Estimated avoided loss had it ruptured: $8M-$22M "
    "depending on environmental sensitivity of the rupture location. Loss "
    "control engineer notes the inspection cycle that caught this was newly "
    "tightened (every 18 months instead of 36) following a 2024 NTSB advisory "
    "on similar pipelines. Cascade has implemented the recommendations promptly. "
    "Energy book is $4.8M premium with two other transmission-line accounts "
    "on older inspection cycles.\n\n"
    "Compliance framework requires that quarterly portfolio reviews address: "
    "(a) loss ratio trends by line and segment versus targets, (b) "
    "concentration risks at the vendor, sub-segment, and geographic level, "
    "(c) any pending regulatory matters that could affect coverage outcomes, "
    "(d) action items from prior reviews that remain open, and (e) "
    "recommendations for the upcoming renewal cycle including any accounts "
    "flagged for non-renewal or restructuring.\n\n"
    "Task: produce a portfolio-level memorandum following the compliance "
    "framework above. Identify the top three concentration risks, recommend "
    "concrete actions with owners and dates, and flag any of the four cases "
    "where coverage outcome could materially affect the portfolio loss ratio. "
    "Be specific about which prior action items remain open and what should "
    "be reprioritized. Conclude with a summary table of recommended posture "
    "by account."
)


def pick_prompt(isl):
    if isl <= 1024:
        body = PROMPT_512
    elif isl <= 3072:
        body = PROMPT_2048
    else:
        body = PROMPT_4096
    if USE_SHARED_PREFIX:
        return SHARED_PREFIX + body
    return body


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip().splitlines()
        if not out:
            return {"name": "unknown", "memory_mb": 0, "util_pct": 0}
        name, mem, util = [p.strip() for p in out[0].split(",")]
        return {"name": name, "memory_mb": int(mem), "util_pct": int(util)}
    except Exception:
        return {"name": "unknown", "memory_mb": 0, "util_pct": 0}


async def one_request(session, osl, prompt, results, fails):
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    auth = BASIC_AUTH
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    try:
        async with session.post(ENDPOINT, json=payload, headers=headers, auth=auth) as resp:
            if resp.status != 200:
                fails.append(resp.status)
                return
            async for raw in resp.content:
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or choices[0].get("delta", {}).get("content") or ""
                if text and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0
            if ttft is None:
                fails.append("no_tokens")
                return
            results.append({"ttft_ms": ttft, "total_ms": total_ms, "tokens": osl})
    except Exception as e:
        fails.append(repr(e))


async def worker(session, osl, isl, results, fails, stop_at):
    while time.perf_counter() < stop_at:
        prompt = pick_prompt(isl)
        await one_request(session, osl, prompt, results, fails)


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


async def main():
    global ENDPOINT, MODEL, TOKEN, BASIC_AUTH, USE_SHARED_PREFIX

    ap = argparse.ArgumentParser(description="LLM inference benchmark")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--isl", type=int, default=512)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--duration", type=int, default=90)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runtime", default="vllm")
    ap.add_argument("--token", default="")
    ap.add_argument("--basic-auth", default="", metavar="USER:PASS",
                    help="HTTP Basic Auth credentials in user:pass format")
    ap.add_argument("--chunked-prefill", action="store_true")
    ap.add_argument("--shared-prefix", action="store_true")
    ap.add_argument("--output-dir", default="./results/real")
    args = ap.parse_args()

    ENDPOINT = args.endpoint
    MODEL = args.model
    TOKEN = args.token
    USE_SHARED_PREFIX = args.shared_prefix
    if args.basic_auth:
        user, _, password = args.basic_auth.partition(":")
        BASIC_AUTH = aiohttp.BasicAuth(user, password)

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("tag={}  isl={}  osl={}  concurrency={}  duration={}s".format(
        args.tag, args.isl, args.osl, args.concurrency, args.duration))
    print("endpoint={}  model={}  runtime={}".format(ENDPOINT, MODEL, args.runtime))

    results = []
    fails = []
    stop_at = time.perf_counter() + args.duration
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    conn = aiohttp.TCPConnector(limit=0)
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_start = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        workers = [worker(session, args.osl, args.isl, results, fails, stop_at)
                   for _ in range(args.concurrency)]
        await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.perf_counter() - t_start
    ttft_vals = [r["ttft_ms"] for r in results]
    total_vals = [r["total_ms"] for r in results]
    tokens_total = sum(r["tokens"] for r in results)
    throughput_tok = tokens_total / elapsed if elapsed > 0 else 0.0
    throughput_req = len(results) / elapsed if elapsed > 0 else 0.0

    metrics = {
        "ttft_ms": {
            "p50": round(percentile(ttft_vals, 50), 1),
            "p90": round(percentile(ttft_vals, 90), 1),
            "p95": round(percentile(ttft_vals, 95), 1),
            "p99": round(percentile(ttft_vals, 99), 1),
            "mean": round(statistics.mean(ttft_vals), 1) if ttft_vals else 0.0,
        },
        "total_latency_ms": {
            "p50": round(percentile(total_vals, 50), 1),
            "p95": round(percentile(total_vals, 95), 1),
            "p99": round(percentile(total_vals, 99), 1),
        },
        "throughput_tokens_per_sec": round(throughput_tok, 1),
        "throughput_req_per_sec": round(throughput_req, 2),
        "total_requests": len(results) + len(fails),
        "successful_requests": len(results),
        "failed_requests": len(fails),
    }

    out = {
        "meta": {
            "tag": args.tag,
            "runtime": args.runtime,
            "model": args.model,
            "gpu": gpu_info(),
            "config": {
                "chunked_prefill": bool(args.chunked_prefill),
                "tensor_parallel_size": 1,
                "shared_prefix": bool(args.shared_prefix),
            },
            "workload": {
                "isl_approx": args.isl,
                "osl_max": args.osl,
                "concurrency": args.concurrency,
                "duration_secs": args.duration,
            },
            "synthetic": False,
            "timestamp": started_iso,
        },
        "metrics": metrics,
    }

    print("TTFT ms  p50={p50}  p95={p95}  p99={p99}  mean={mean}".format(**metrics["ttft_ms"]))
    print("throughput  {} tok/s  {} req/s".format(
        metrics["throughput_tokens_per_sec"], metrics["throughput_req_per_sec"]))
    print("requests  total={}  failed={}".format(
        metrics["total_requests"], metrics["failed_requests"]))

    out_path = os.path.join(output_dir, args.tag + ".json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    random.seed(0)
    asyncio.run(main())
