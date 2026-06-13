"""
planner/compare.py — compare N capacity scenarios on cost, safety, and latency.

Returns labeled winners across three dimensions:
  cheapest      — lowest daily GPU cost (replicas proxy when no pricing data)
  safest        — highest confidence × narrowest uncertainty band
  best_latency  — lowest predicted TTFT p50

Also surfaces structural notes when GPU choices differ in KV budget
(e.g. H200 vs H100 for long-context concurrency-bound workloads).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from planner.capacity import CapacityEstimate
from planner.cost import CostEstimate


_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class ComparisonEntry:
    label: str
    estimate: CapacityEstimate
    cost: Optional[CostEstimate] = None
    gpu_name: str = ""
    model_name: str = ""
    dtype: str = ""


@dataclass
class Winner:
    dimension: str    # "cheapest" | "safest" | "best_latency"
    entry_label: str
    metric_value: str  # human-readable, e.g. "$12.50/day" or "120 ms TTFT"
    tradeoff: str      # what this option gives up vs the alternatives


@dataclass
class ComparisonResult:
    entries: List[ComparisonEntry]
    cheapest: Winner
    safest: Winner
    best_latency: Winner
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare(entries: List[ComparisonEntry]) -> ComparisonResult:
    """Compare two or more capacity scenarios and label the winner per dimension."""
    if len(entries) < 2:
        raise ValueError(
            f"compare() requires at least 2 entries, got {len(entries)}."
        )

    return ComparisonResult(
        entries=entries,
        cheapest=_find_cheapest(entries),
        safest=_find_safest(entries),
        best_latency=_find_best_latency(entries),
        notes=_generate_notes(entries),
    )


# ---------------------------------------------------------------------------
# Dimension scoring helpers
# ---------------------------------------------------------------------------


def _cost_per_day(entry: ComparisonEntry) -> float:
    if entry.cost is not None:
        return entry.cost.on_demand.cost_day_usd
    return float(entry.estimate.replicas)   # replica count as cost proxy


def _safety_score(entry: ComparisonEntry) -> float:
    conf_rank = _CONF_RANK.get(entry.estimate.confidence, 1)
    est = entry.estimate
    band = (est.replicas_high - est.replicas_low) / max(est.replicas, 1)
    # Higher confidence dominates; narrower band breaks ties at same confidence.
    return conf_rank * 10.0 - band


def _find_cheapest(entries: List[ComparisonEntry]) -> Winner:
    best = min(entries, key=_cost_per_day)
    others = [e for e in entries if e is not best]

    if best.cost is not None:
        metric = f"${best.cost.on_demand.cost_day_usd:.2f}/day on-demand"
        others_str = ", ".join(
            f"{e.label} (${_cost_per_day(e):.2f}/day)" for e in others
        )
        tradeoff = (
            f"Saves vs {others_str}. "
            f"Confidence: {best.estimate.confidence.upper()}. "
            f"Replicas: {best.estimate.replicas} "
            f"({best.estimate.replicas_low}–{best.estimate.replicas_high})."
        )
    else:
        metric = f"{best.estimate.replicas} replicas (no pricing data)"
        others_str = ", ".join(f"{e.label} ({e.estimate.replicas} replicas)" for e in others)
        tradeoff = (
            f"Fewest replicas vs {others_str}. "
            "No pricing data — add to catalog/costs.yaml for dollar figures."
        )

    return Winner(dimension="cheapest", entry_label=best.label,
                  metric_value=metric, tradeoff=tradeoff)


def _find_safest(entries: List[ComparisonEntry]) -> Winner:
    best = max(entries, key=_safety_score)
    others = [e for e in entries if e is not best]
    est = best.estimate

    band_pct = round(
        (est.replicas_high - est.replicas_low) / max(est.replicas, 1) * 100
    )
    metric = f"{est.confidence.upper()} confidence, ±{band_pct}% band"

    lower_conf = [
        e for e in others
        if _CONF_RANK.get(e.estimate.confidence, 1)
        < _CONF_RANK.get(est.confidence, 1)
    ]
    if lower_conf:
        worse_str = ", ".join(
            f"{e.label} ({e.estimate.confidence.upper()})" for e in lower_conf
        )
        tradeoff = (
            f"Highest confidence — alternatives with lower confidence: {worse_str}. "
            f"Replica range: {est.replicas_low}–{est.replicas_high}."
        )
    else:
        others_str = ", ".join(
            f"{e.label} ({e.estimate.replicas_low}–{e.estimate.replicas_high})" for e in others
        )
        tradeoff = (
            f"All options share {est.confidence.upper()} confidence. "
            f"{best.label} has the narrowest band "
            f"({est.replicas_low}–{est.replicas_high}) vs {others_str}."
        )

    return Winner(dimension="safest", entry_label=best.label,
                  metric_value=metric, tradeoff=tradeoff)


def _find_best_latency(entries: List[ComparisonEntry]) -> Winner:
    best = min(entries, key=lambda e: e.estimate.ttft_estimate.ttft_ms)
    others = [e for e in entries if e is not best]

    ttft = best.estimate.ttft_estimate.ttft_ms
    metric = f"{ttft:.0f} ms predicted TTFT p50"

    others_str = ", ".join(
        f"{e.label} ({e.estimate.ttft_estimate.ttft_ms:.0f} ms)" for e in others
    )
    tradeoff = (
        f"Lowest predicted TTFT vs {others_str}. "
        f"Binding constraint: {best.estimate.binding_constraint}. "
        "Caveat: TTFT uses M/M/1 queuing heuristic — validate on a live GPU."
    )

    return Winner(dimension="best_latency", entry_label=best.label,
                  metric_value=metric, tradeoff=tradeoff)


def _generate_notes(entries: List[ComparisonEntry]) -> List[str]:
    notes: List[str] = []

    # H200 vs H100: surface the KV-budget concurrency advantage
    gpus = {e.gpu_name for e in entries}
    if "h200_sxm" in gpus and "h100_sxm" in gpus:
        h200 = next((e for e in entries if e.gpu_name == "h200_sxm"), None)
        h100 = next((e for e in entries if e.gpu_name == "h100_sxm"), None)
        if h200 and h100:
            h200_seqs = h200.estimate.kv_budget.max_concurrent_seqs
            h100_seqs = h100.estimate.kv_budget.max_concurrent_seqs
            if h200_seqs > h100_seqs:
                notes.append(
                    f"H200 KV-cache advantage: {h200_seqs} vs {h100_seqs} max concurrent seqs "
                    f"(141 GB HBM3e vs 80 GB). "
                    "For long-context or concurrency-bound workloads H200 often needs fewer "
                    "replicas despite its higher $/hr."
                )

    # Warn on LOW confidence entries
    low = [e for e in entries if e.estimate.confidence == "low"]
    if low:
        labels = ", ".join(e.label for e in low)
        notes.append(
            f"{labels}: LOW confidence — no benchmark anchor. "
            "Run POST /benchmarks/run to calibrate before committing infrastructure."
        )

    return notes
