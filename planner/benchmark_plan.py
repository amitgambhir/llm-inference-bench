"""
planner/benchmark_plan.py — generate an ordered run_bench.py test matrix
from a CapacityEstimate.

The matrix is ordered so the FIRST test collapses the widest confidence gap:
  - LOW confidence, prefill-bound  → single-replica saturation sweep at real ISL
  - LOW confidence, decode-bound   → single-replica decode sweep (batch ramp)
  - MEDIUM confidence              → workload-shape confirmation first
  - HIGH confidence                → scale validation is the remaining unknown

Every item carries:
  command              : ready-to-run run_bench.py invocation
  purpose              : one-line human description
  collapses_confidence_on : what assumption this test validates
  priority             : 1 = run first (highest confidence gain)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from planner.capacity import CapacityEstimate

# Default saturation-sweep concurrency levels for single-replica tests
_SATURATION_CONCURRENCIES = [16, 32, 64]

# Burst / soak durations (seconds)
_BURST_DURATION_S = 600    # 10 min
_SOAK_DURATION_S = 7200    # 2 h


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkStep:
    priority: int
    label: str
    command: str
    purpose: str
    collapses_confidence_on: str


@dataclass
class BenchmarkPlan:
    steps: list[BenchmarkStep]
    model_name: str
    gpu_name: str
    confidence: str
    binding_constraint: str
    rationale: str          # one paragraph on why this ordering was chosen


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def _cmd(
    model: str,
    isl: int,
    osl: int,
    concurrency: int,
    tag: str,
    duration: int = 90,
    chunked_prefill: bool = False,
) -> str:
    parts = [
        "python collect/run_bench.py",
        f"--model {model}",
        f"--isl {isl}",
        f"--osl {osl}",
        f"--concurrency {concurrency}",
        f"--duration {duration}",
        f"--tag {tag}",
    ]
    if chunked_prefill:
        parts.append("--chunked-prefill")
    return " ".join(parts)


def _safe_tag(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace(".", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def benchmark_plan(
    estimate: CapacityEstimate,
    model_name: str,
    gpu_name: str,
    endpoint: str = "http://localhost:8000",
    include_chunked_prefill_variant: bool = True,
) -> BenchmarkPlan:
    """Generate an ordered test matrix for the given CapacityEstimate.

    The ordering principle: first test that maximally collapses the confidence gap.
    For a LOW-confidence, prefill-bound estimate the single-replica ISL sweep at
    the real workload ISL directly measures the MFU assumption the estimate rested on.
    """
    t = estimate.traffic
    isl = _infer_isl(estimate)
    osl = _infer_osl(estimate)
    replicas = estimate.replicas
    confidence = estimate.confidence
    constraint = estimate.binding_constraint
    model_tag = _safe_tag(model_name)

    steps: list[BenchmarkStep] = []
    priority = 1

    # ── TEST GROUP 1: single-replica saturation sweep ─────────────────────
    # Ordered FIRST for LOW confidence — directly measures the prefill/decode
    # ceiling the estimate assumed. Run at the real workload ISL so the measured
    # MFU matches the scenario's compute regime.
    for c in _SATURATION_CONCURRENCIES:
        tag = f"{model_tag}_{gpu_name}_isl{isl}_c{c}_sat"
        steps.append(BenchmarkStep(
            priority=priority,
            label=f"Single-replica saturation @ c={c}",
            command=_cmd(model_name, isl, osl, c, tag),
            purpose=(
                f"Find the single-replica saturation point by ramping concurrency to {c}. "
                f"Measures real {'prefill' if 'prefill' in constraint else 'decode'} ceiling at ISL={isl}."
            ),
            collapses_confidence_on=(
                f"MFU assumption (currently {estimate.mfu_used:.0%}) for prefill ceiling; "
                "upgrades confidence from LOW→MEDIUM/HIGH for this (model, gpu, dtype, ISL) band."
            ),
        ))
        priority += 1

    # ── TEST GROUP 2: workload-shape confirmation ─────────────────────────
    # Run at real ISL/OSL to confirm the prefill-vs-decode ratio assumed by
    # size_replicas. Critical for prefill-bound scenarios to validate that the
    # binding constraint is correctly identified.
    tag = f"{model_tag}_{gpu_name}_isl{isl}_osl{osl}_c32_shape"
    steps.append(BenchmarkStep(
        priority=priority,
        label="Workload-shape confirmation @ c=32",
        command=_cmd(model_name, isl, osl, 32, tag),
        purpose=(
            f"Confirm binding constraint ({constraint}) at the real workload shape "
            f"(ISL={isl}, OSL={osl}). Validates that prefill : decode ratio assumed "
            f"by the replica calculation holds on real hardware."
        ),
        collapses_confidence_on=(
            f"Binding constraint: confirms whether '{constraint}' holds at "
            f"ISL={isl}/OSL={osl}, and validates bw_eff assumption "
            f"(currently {estimate.bw_eff_used:.0%}) for decode."
        ),
    ))
    priority += 1

    # Optional: chunked-prefill variant at high ISL — only meaningful if ISL is large
    if include_chunked_prefill_variant and isl >= 4096:
        tag_cp = f"{model_tag}_{gpu_name}_isl{isl}_osl{osl}_c32_chunked"
        steps.append(BenchmarkStep(
            priority=priority,
            label=f"Chunked-prefill variant @ c=32, ISL={isl}",
            command=_cmd(model_name, isl, osl, 32, tag_cp, chunked_prefill=True),
            purpose=(
                f"Compare chunked-prefill on vs off at ISL={isl}. "
                "Quantifies TTFT benefit (if any) vs the baseline run above."
            ),
            collapses_confidence_on=(
                "TTFT queuing model: validates whether chunked prefill materially "
                "changes the TTFT at this ISL. Required before --enable-chunked-prefill "
                "is added to the serving config."
            ),
        ))
        priority += 1

    # ── TEST GROUP 3: horizontal scale validation ─────────────────────────
    # Validate that N replicas actually deliver N× throughput.
    # Run at peak_rps to exercise the full fleet.
    peak_rps_rounded = max(1, round(t.peak_rps))
    # Approximate concurrency for the scale test: Little's Law at peak
    # use a conservatively large concurrency to stress the fleet
    scale_concurrency = min(256, replicas * 32)
    tag = f"{model_tag}_{gpu_name}_r{replicas}_scale"
    steps.append(BenchmarkStep(
        priority=priority,
        label=f"Horizontal scale @ {replicas} replicas",
        command=_cmd(model_name, isl, osl, scale_concurrency, tag, duration=300),
        purpose=(
            f"Validate linear scaling: {replicas} replicas should sustain "
            f"~{peak_rps_rounded} req/s at ISL={isl}. "
            f"Run at concurrency={scale_concurrency} for 5 min to expose "
            "load-balancer skew, cold-start lag, and memory pressure."
        ),
        collapses_confidence_on=(
            "Horizontal scaling assumption: confirms that {r} replicas provide "
            "{r}× single-replica throughput (no load-balancer hotspots, "
            "no KV-cache eviction at fleet scale).".format(r=replicas)
        ),
    ))
    priority += 1

    # ── TEST GROUP 4: burst test ──────────────────────────────────────────
    # 10-min peak load — reveals autoscaling lag and cold-start overhead.
    burst_concurrency = min(512, scale_concurrency * 2)
    tag = f"{model_tag}_{gpu_name}_burst"
    steps.append(BenchmarkStep(
        priority=priority,
        label="Burst test (10 min at peak)",
        command=_cmd(model_name, isl, osl, burst_concurrency, tag, duration=_BURST_DURATION_S),
        purpose=(
            f"Sustain {burst_concurrency} concurrent requests for 10 min. "
            "Exposes autoscaling lag, cold-start overhead, and KV-cache "
            "eviction spikes that short runs miss."
        ),
        collapses_confidence_on=(
            "TTFT SLO under sustained peak: the M/M/1 queuing model used by "
            "ttft_estimate is a steady-state approximation — this test validates "
            "whether the queue actually stabilises or keeps growing."
        ),
    ))
    priority += 1

    # ── TEST GROUP 5: soak test ───────────────────────────────────────────
    # 2-h average load — validates stability and memory growth.
    avg_concurrency = max(8, math.ceil(t.avg_rps * 2))   # rough steady-state concurrency
    tag = f"{model_tag}_{gpu_name}_soak"
    steps.append(BenchmarkStep(
        priority=priority,
        label="Soak test (2 h at avg load)",
        command=_cmd(model_name, isl, osl, avg_concurrency, tag, duration=_SOAK_DURATION_S),
        purpose=(
            f"Run at average load ({avg_concurrency} concurrent) for 2 hours. "
            "Validates memory stability (no KV-cache fragmentation growth), "
            "throughput consistency, and absence of latency drift over time."
        ),
        collapses_confidence_on=(
            "Long-run stability: short benchmarks cannot detect memory leaks, "
            "KV-cache fragmentation, or throughput degradation that emerges "
            "over hours of continuous operation."
        ),
    ))

    # ── Rationale paragraph ───────────────────────────────────────────────
    if confidence == "low":
        if "prefill" in constraint:
            rationale = (
                f"Confidence is LOW and the estimate is {constraint}. The largest uncertainty "
                f"is the prefill MFU assumption ({estimate.mfu_used:.0%} GPU default, not calibrated). "
                f"The saturation sweep at ISL={isl} is ordered first: it directly measures the "
                f"prefill ceiling and back-computes real MFU, upgrading confidence for this scenario. "
                f"Run tests 1–3 before sizing infrastructure."
            )
        else:
            rationale = (
                f"Confidence is LOW and the estimate is {constraint}. The largest uncertainty "
                f"is the bandwidth efficiency assumption ({estimate.bw_eff_used:.0%} GPU default). "
                f"The saturation sweep at ISL={isl} is ordered first to calibrate both prefill MFU "
                f"and decode bandwidth efficiency before committing to a replica count."
            )
    elif confidence == "medium":
        rationale = (
            f"Confidence is MEDIUM ({constraint}). Anchors exist but the scenario's "
            f"ISL/concurrency is extrapolated. The saturation sweep at ISL={isl} "
            f"fills the gap directly; the workload-shape test confirms the binding constraint. "
            f"Run tests 1–4 before production cutover."
        )
    else:
        rationale = (
            f"Confidence is HIGH ({constraint}). Anchors are calibrated near this scenario. "
            f"The saturation sweep validates nothing new for single-replica sizing; the scale "
            f"test (test {len(_SATURATION_CONCURRENCIES) + 2}) is the most valuable remaining run "
            f"to confirm that {replicas} replicas deliver the expected aggregate throughput."
        )

    return BenchmarkPlan(
        steps=steps,
        model_name=model_name,
        gpu_name=gpu_name,
        confidence=confidence,
        binding_constraint=constraint,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Helpers to extract isl/osl from CapacityEstimate
# ---------------------------------------------------------------------------


def _infer_isl(estimate: CapacityEstimate) -> int:
    """Back-compute ISL from traffic: avg_rps * isl = input_tps_avg."""
    avg_rps = estimate.traffic.avg_rps
    if avg_rps > 0:
        return max(1, round(estimate.traffic.input_tps_avg / avg_rps))
    return 512


def _infer_osl(estimate: CapacityEstimate) -> int:
    """Back-compute OSL from traffic: avg_rps * osl = output_tps_avg."""
    avg_rps = estimate.traffic.avg_rps
    if avg_rps > 0:
        return max(1, round(estimate.traffic.output_tps_avg / avg_rps))
    return 128


# ---------------------------------------------------------------------------
# Human-readable render
# ---------------------------------------------------------------------------


def render_plan(plan: BenchmarkPlan) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║              Benchmark Plan                              ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"  Model      : {plan.model_name}",
        f"  GPU        : {plan.gpu_name}",
        f"  Confidence : {plan.confidence.upper()}  |  Constraint: {plan.binding_constraint}",
        "",
        f"  Ordering rationale:",
        *[f"    {line}" for line in plan.rationale.split(". ") if line],
        "",
    ]
    for step in plan.steps:
        lines += [
            f"── Test {step.priority}: {step.label}",
            f"   Purpose  : {step.purpose}",
            f"   Validates: {step.collapses_confidence_on}",
            f"   Command  :",
            f"     {step.command}",
            "",
        ]
    return "\n".join(lines)
