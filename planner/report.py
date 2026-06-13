"""
planner/report.py — generate a shareable Markdown recommendation report.

render_report() wraps a CapacityEstimate + optional CostEstimate into a
Markdown document suitable for export.  Always stamps the mode badge so
readers know whether figures come from a pure estimate, partial validation,
or a fully calibrated benchmark run.

Mode badges:
  estimate_only         — pure roofline, no live GPU data
  partially_validated   — some benchmark steps completed
  validated_by_benchmark — calibrated from a real run
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from planner.capacity import CapacityEstimate
from planner.cost import CostEstimate


_MODE_LABELS = {
    "estimate_only": "⚠ ESTIMATE ONLY — not validated by benchmark",
    "partially_validated": "◑ PARTIALLY VALIDATED — some benchmark steps run",
    "validated_by_benchmark": "✓ VALIDATED BY BENCHMARK",
}

_CONF_BAND = {
    "high": "±10%",
    "medium": "±25%",
    "low": "±50%",
}


def render_report(
    estimate: CapacityEstimate,
    scenario_label: str,
    model_name: str,
    gpu_name: str,
    dtype: str,
    isl: int,
    osl: int,
    requests_per_day: int,
    peak_multiplier: float,
    tp: int = 1,
    traffic_class: str = "realtime",
    cost: Optional[CostEstimate] = None,
    anchors_used: Optional[List[str]] = None,
    mode: str = "estimate_only",
    generated_at: Optional[datetime] = None,
) -> str:
    """Return a Markdown report string.

    Args:
        estimate:          Sized CapacityEstimate from planner.capacity.plan().
        scenario_label:    Human-readable scenario name (e.g. "h100-bf16-chat").
        model_name:        Catalog model key or display name.
        gpu_name:          Catalog GPU key.
        dtype:             Serving dtype (e.g. "fp8", "bf16").
        isl:               Input sequence length (tokens).
        osl:               Output sequence length (tokens).
        requests_per_day:  Daily request volume.
        peak_multiplier:   Peak burst multiplier over avg.
        tp:                Tensor-parallel degree.
        traffic_class:     "realtime" | "mixed" | "batch".
        cost:              Optional CostEstimate from planner.cost.compute_cost().
        anchors_used:      Human-readable anchor descriptions for evidence section.
        mode:              "estimate_only" | "partially_validated" | "validated_by_benchmark".
        generated_at:      Override report timestamp (default: UTC now).
    """
    ts = generated_at or datetime.now(timezone.utc)
    mode_label = _MODE_LABELS.get(mode, _MODE_LABELS["estimate_only"])
    conf = estimate.confidence
    band = _CONF_BAND.get(conf, "±?%")
    est = estimate

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"# Capacity Recommendation: {scenario_label}",
        "",
        f"> **{mode_label}**",
        "",
        f"Generated: {ts.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # ── Workload summary ──────────────────────────────────────────────────
    lines += [
        "## Workload",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Model | `{model_name}` |",
        f"| GPU | `{gpu_name}` |",
        f"| Dtype | `{dtype}` |",
        f"| Tensor parallel | `{tp}` |",
        f"| Traffic class | `{traffic_class}` |",
        f"| Requests/day | `{requests_per_day:,}` |",
        f"| Peak multiplier | `{peak_multiplier}×` |",
        f"| ISL | `{isl}` tokens |",
        f"| OSL | `{osl}` tokens |",
        f"| Avg RPS | `{est.traffic.avg_rps:.1f}` |",
        f"| Peak RPS | `{est.traffic.peak_rps:.1f}` |",
        "",
    ]

    # ── Sizing recommendation ─────────────────────────────────────────────
    lines += [
        "## Sizing Recommendation",
        "",
        f"| | Value |",
        f"|---|---|",
        f"| **Replicas** | **{est.replicas}** |",
        f"| Range ({conf.upper()}, {band}) | {est.replicas_low} – {est.replicas_high} |",
        f"| Binding constraint | `{est.binding_constraint}` |",
        f"| Confidence | `{conf.upper()}` |",
        f"| Max concurrent seqs/replica | `{est.kv_budget.max_concurrent_seqs}` |",
        f"| Prefill ceiling (per GPU) | `{est.prefill_tps_gpu:,.0f}` tok/s |",
        f"| Decode ceiling (per GPU) | `{est.decode_tps_gpu:,.0f}` tok/s |",
        "",
    ]

    # TTFT estimate
    ttft = est.ttft_estimate
    slo_flag = "✓ SLO met" if ttft.slo_met else f"✗ SLO BREACHED — {ttft.slo_breach_reason}"
    lines += [
        "### Latency Estimate",
        "",
        f"| | |",
        f"|---|---|",
        f"| Predicted TTFT p50 | `{ttft.ttft_ms:.0f} ms` |",
        f"| TTFT compute | `{ttft.ttft_compute_ms:.0f} ms` |",
        f"| TTFT queue wait (M/M/1) | `{ttft.ttft_queue_ms:.0f} ms` |",
        f"| SLO ({ttft.slo_ms:.0f} ms) | {slo_flag} |",
        "",
        "> Queue wait is a heuristic — validate on a live GPU before committing.",
        "",
    ]

    # ── Cost envelope (optional) ──────────────────────────────────────────
    if cost is not None:
        od = cost.on_demand
        rv = cost.reserved
        lines += [
            "## Cost Envelope",
            "",
            f"| | On-demand | Reserved |",
            f"|---|---|---|",
            f"| GPU-hours/day | `{od.gpu_hours_day:.1f}` | `{rv.gpu_hours_day:.1f}` |",
            f"| $/day | `${od.cost_day_usd:.2f}` | `${rv.cost_day_usd:.2f}` |",
            f"| $/month (30d) | `${od.cost_month_usd:.2f}` | `${rv.cost_month_usd:.2f}` |",
            f"| $/1M tokens | `${od.cost_per_1m_tokens_usd:.4f}` | `${rv.cost_per_1m_tokens_usd:.4f}` |",
            f"| $/request | `${od.cost_per_request_usd:.6f}` | `${rv.cost_per_request_usd:.6f}` |",
            "",
            f"Prefill/decode split: "
            f"{cost.breakdown.prefill_fraction:.0%} prefill / "
            f"{cost.breakdown.decode_fraction:.0%} decode.",
            "",
        ]

    # ── Benchmark evidence ────────────────────────────────────────────────
    lines += ["## Benchmark Evidence", ""]
    if mode == "estimate_only":
        lines += [
            "**No benchmark data available.** Figures are pure roofline estimates "
            "using GPU-default MFU and bandwidth efficiency.",
            "",
            "Run `POST /benchmarks/run` to generate calibration data, then "
            "`POST /ingest/{run_id}` to update confidence.",
            "",
        ]
    elif anchors_used:
        lines += [
            f"Confidence calibrated from {len(anchors_used)} anchor(s):",
            "",
        ]
        for a in anchors_used:
            lines.append(f"- {a}")
        lines.append("")
    else:
        lines += [
            "Benchmark runs completed but no anchor details provided. "
            "Re-run with anchors_used populated for full evidence chain.",
            "",
        ]

    # ── Assumptions ───────────────────────────────────────────────────────
    if est.assumptions:
        lines += ["## Assumptions", ""]
        for a in est.assumptions:
            lines.append(f"- {a}")
        lines.append("")

    # ── Warnings / risks ─────────────────────────────────────────────────
    if est.warnings:
        lines += ["## Warnings", ""]
        for w in est.warnings:
            lines.append(f"> ⚠ {w}")
        lines.append("")

    # ── Next validation steps ─────────────────────────────────────────────
    lines += ["## Next Steps", ""]
    if mode == "estimate_only":
        lines += [
            "1. Run the generated benchmark plan (`POST /benchmark-plan`) to get "
            "calibration commands.",
            "2. Execute saturation sweep at ISL={isl} to measure real MFU.".format(isl=isl),
            "3. Ingest results with `POST /ingest/{run_id}` to upgrade confidence.",
            "4. Re-run estimate — confidence should improve to MEDIUM or HIGH.",
        ]
    elif mode == "partially_validated":
        lines += [
            "1. Complete remaining benchmark steps (soak + burst tests).",
            "2. Ingest all completed runs to fully calibrate the confidence band.",
            "3. Share this report with infrastructure team for capacity pre-allocation.",
        ]
    else:
        lines += [
            "1. ✓ Benchmark evidence is in place — share this report.",
            "2. Re-validate after model updates or ISL distribution changes.",
            "3. Monitor live TTFT p95 against the SLO — queue heuristic may diverge "
            "under non-Poisson arrival patterns.",
        ]

    lines.append("")
    lines.append(
        f"*Report generated by llm-inference-bench · "
        f"confidence: {conf.upper()} · mode: {mode}*"
    )

    return "\n".join(lines)
